from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener
from miniDSL.miniDSLLexer import miniDSLLexer
from miniDSL.miniDSLParser import miniDSLParser
from miniDSL.miniDSLVisitor import miniDSLVisitor

"""
Check semantic errors with several manipulated testing. Won't fix automatically.
Hard coding.

Returns: (T/F, err_messages)
"""


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
            "*": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            "/": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            "<": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            "<=": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            ">": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            ">=": {
                ("PolyExp", "PolyExp"),
                ("Neuron", "PolyExp"),
                ("PolyExp", "Neuron"),
            },
            # TODO: expand this
        }
        self.valid_metadata = {
            "weight",
            "bias",
            "equations",
            "layer",
            "l",
            "u",
            "L",
            "U",
        }

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

    # Check binary operations (e.g. PolyExp * PolyExp)
    def visitBinopExp(self, ctx):
        left_type = self.infer_type(ctx.getChild(0))
        right_type = self.infer_type(ctx.getChild(2))
        print(left_type)
        print(right_type)
        op = ctx.getChild(1).getText()
        if (left_type, right_type) in self.invalid_type_pairs.get(op, set()):
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid type combination: {left_type} {op} {right_type}"
            )
        return self.visitChildren(ctx)

    def infer_type(self, ctx):
        text = ctx.getText()

        if hasattr(ctx, "VAR"):
            return self.var_types.get(text, "Unknown")

        if any(
            text.startswith(prefix + "[")
            and any(
                k in text
                for k in [
                    "L",
                    "U",
                ]
            )
            for prefix in ["curr", "prev", "curr_list", "prev_0", "prev_1"]
        ):
            return "PolyExp"

        if any(
            text.startswith(prefix + "[")
            and any(
                k in text
                for k in [
                    "l",
                    "u",
                ]
            )
            for prefix in ["curr", "prev", "curr_list", "prev_0", "prev_1"]
        ):
            return "Float"

        if text in {"curr", "prev", "curr_list", "prev_0", "prev_1"}:
            return "Neuron"

        if text.isdigit():
            return "Int"
        if "." in text and text.replace(".", "", 1).isdigit():
            return "Float"

        # Binary expression: recursively infer children types
        if ctx.getChildCount() == 3:
            left_ctx = ctx.getChild(0)
            op = ctx.getChild(1).getText()
            right_ctx = ctx.getChild(2)
            left_type = self.infer_type(left_ctx)
            right_type = self.infer_type(right_ctx)

            return "PolyExp"

        return "Unknown"

    def report(self):
        if not self.errors:
            print("✅ No semantic errors found.")
        else:
            for err in self.errors:
                print("❌", err)


def check_semantic(dsl):
    input_stream = InputStream(dsl)
    lexer = miniDSLLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = miniDSLParser(token_stream)

    error_listener = SemanticErrorListener()
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)

    try:
        tree = parser.transformer()
    except Exception as e:
        return False, [f"[Parse Error] {str(e)}"]

    if error_listener.errors:
        return False, error_listener.errors

    checker = SemanticChecker()
    checker.visit(tree)

    if checker.errors:
        return False, checker.errors

    return True, dsl, []


if __name__ == "__main__":
    dsl = """
transformer deeppoly{
    HardTanh ->
        (prev[l] >= 1) ? (1, 1, 1, 1)
        : (prev[u] <= -1) ? (-1, -1, -1, -1)
        : (prev[l] >= -1 and prev[u] <= 1) ? (prev[l], prev[u], prev, prev)
        : (prev[l] < -1 and prev[u] > 1) ? (-1, 1, prev, prev)
        : (prev[l] < -1) ?
            (-1, min(prev[u], 1),
             prev*(min(prev[u], 1)-(-1))/(prev[u] - prev[l]) - ((2*min(prev[u], 1)*(-1))/(prev[u]-prev[l])),
             prev*(min(prev[u], 1)-(-1))/(prev[u] - prev[l]) - ((2*min(prev[u], 1)*(-1))/(prev[u]-prev[l])))
        : (max(prev[l], -1), 1,
           prev*(1-max(prev[l], -1))/(prev[u]-prev[l]) - ((2*1*max(prev[l], -1))/(prev[u]-prev[l])),
           prev*(1-max(prev[l], -1))/(prev[u]-prev[l]) - ((2*1*max(prev[l], -1))/(prev[u]-prev[l])))
        ;
}
    """
    result, dsl, errs = check_semantic(dsl)
    print(result)

    for e in errs:
        print(e)
