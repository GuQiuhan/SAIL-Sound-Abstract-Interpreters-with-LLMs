from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener
from miniDSL.miniDSLLexer import miniDSLLexer
from miniDSL.miniDSLParser import miniDSLParser
from miniDSL.miniDSLVisitor import miniDSLVisitor


class SemanticErrorListener(ErrorListener):
    def __init__(self):
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"[Syntax Error] Line {line}:{column} {msg}")


class SemanticChecker(miniDSLVisitor):
    def __init__(self):
        self.defined_vars = set()
        self.var_types = dict()
        self.errors = []

        self.valid_funcs = {"sum", "avg", "len", "argmax", "argmin", "max", "min"}
        self.invalid_type_pairs = {
            "*": {("PolyExp", "PolyExp")},
            "/": {("PolyExp", "PolyExp")},
            "<": {("PolyExp", "PolyExp")},
            "<=": {("PolyExp", "PolyExp")},
            ">": {("PolyExp", "PolyExp")},
            ">=": {("PolyExp", "PolyExp")},
            # TODO: expand this
        }
        self.valid_metadata = {"weight", "bias", "equations", "layer"}

    # Register transformer name
    def visitTransstmt(self, ctx):
        name = ctx.transformer().VAR().getText()
        self.defined_vars.add(name)
        return self.visitChildren(ctx)

    # Check undefined variable usage
    def visitVarExp(self, ctx):
        var = ctx.getText()
        if var not in self.defined_vars and var not in {
            "curr",
            "prev",
            "eps",
            "curr_list",
            "prev_0",
            "prev_1",
        }:
            self.errors.append(f"[Line {ctx.start.line}] Undefined variable: {var}")
        return self.visitChildren(ctx)

    # Check for invalid function calls using curry syntax (e.g. max x y)
    def visitCurry(self, ctx):
        func_name = ctx.VAR().getText()
        if func_name not in self.valid_funcs and func_name not in self.defined_vars:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid function call: {func_name}"
            )
        return self.visitChildren(ctx)

    # Check for invalid function calls using func(x, y) style
    def visitFuncCall(self, ctx):
        func_name = ctx.VAR().getText()
        if func_name not in self.valid_funcs and func_name not in self.defined_vars:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid function call: {func_name}"
            )
        return self.visitChildren(ctx)

    # Disallow property-style calls like prev[l].sum
    def visitDot(self, ctx):
        right = ctx.getChild(2).getText()
        if right in {"sum", "avg", "len"}:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid attribute call: use '{right}(expr)' instead of 'expr.{right}'"
            )
        return self.visitChildren(ctx)

    # Check that metadata access uses valid keywords (e.g. curr[weight])
    def visitGetElement(self, ctx):
        base = ctx.expr().getText()
        index = ctx.VAR().getText()
        if base in {"curr", "prev"} and index not in self.valid_metadata:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid metadata access: {base}[{index}]"
            )
        return self.visitChildren(ctx)

    # Enforce type constraints for binary operations (e.g. PolyExp * PolyExp)
    def visitBinopExp(self, ctx):
        left_type = self.infer_type(ctx.getChild(0))
        right_type = self.infer_type(ctx.getChild(2))
        op = ctx.getChild(1).getText()
        if (left_type, right_type) in self.invalid_type_pairs.get(op, set()):
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid type combination: {left_type} {op} {right_type}"
            )
        return self.visitChildren(ctx)

    # Heuristic-based type inference
    def infer_type(self, ctx):
        if hasattr(ctx, "VAR"):
            return self.var_types.get(ctx.getText(), "Unknown")

        text = ctx.getText()
        if text in {"curr", "prev", "curr_list", "prev_0", "prev_1"}:
            return "PolyExp"
        if text.isdigit():
            return "Int"
        if "." in text:
            return "Float"

        if ctx.getChildCount() == 3:
            left = self.infer_type(ctx.getChild(0))
            right = self.infer_type(ctx.getChild(2))
            if left == right:
                return left
        return "Unknown"

    def report(self):
        if not self.errors:
            print("✅ No semantic errors found.")
        else:
            for err in self.errors:
                print("❌", err)


def check_semantic(text):
    input_stream = InputStream(text)
    lexer = miniDSLLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = miniDSLParser(token_stream)

    error_listener = SemanticErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)

    tree = parser.transformer()

    if error_listener.errors:
        for err in error_listener.errors:
            print("❌", err)
        return

    checker = SemanticChecker()
    checker.visit(tree)
    checker.report()


if __name__ == "__main__":
    dsl = """
    transformer T {
        Avgpool -> ((1.0 / curr[size]) * (sum(prev[l])));
    }
    """
    check_semantic(dsl)
