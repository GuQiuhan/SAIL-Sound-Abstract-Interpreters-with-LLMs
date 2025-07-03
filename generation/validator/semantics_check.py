from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener
from miniDSL.miniDSLLexer import miniDSLLexer
from miniDSL.miniDSLParser import miniDSLParser
from miniDSL.miniDSLVisitor import miniDSLVisitor

"""
Check semantic errors with several manipulated testing. Won't fix automatically.
Hard coding.

Returns: (T/F, dsl, err_messages)
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

    def visitTransformer(self, ctx):
        return self.visitChildren(ctx)

    def visitTrans(self, ctx):
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
        print("3")
        func_name = ctx.VAR().getText()
        if func_name not in self.valid_funcs and func_name not in self.defined_vars:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid function call: {func_name}"
            )
        if func_name in {"max", "min"}:
            for child in ctx.expr():
                arg_type = self.infer_type(child)
                if arg_type == "Neuron" or arg_type == "PolyExp":
                    self.errors.append(
                        f"[Line {ctx.start.line}] Invalid use of Neuron in {func_name}: {child.getText()}"
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

    def visitMaxOp(self, ctx):
        op_name = ctx.getChild(0).getText()  # this gets "max" or "min"
        for child in ctx.expr():
            arg_type = self.infer_type(child)
            if arg_type in {"Neuron", "PolyExp"}:
                self.errors.append(
                    f"[Line {ctx.start.line}] Invalid use of {arg_type} in {op_name}: {child.getText()}"
                )
        return self.visitChildren(ctx)

    def visitMaxOpList(self, ctx):
        expr_node = ctx.expr()
        arg_type = self.infer_type(expr_node)
        if arg_type in {"Neuron", "PolyExp"}:
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid use of {arg_type} in {ctx.getChild(0).getText()}: {expr_node.getText()}"
            )
        return self.visitChildren(ctx)

    def visitNeg(self, ctx):
        arg_type = self.infer_type(ctx.expr())
        if arg_type == "Neuron":
            self.errors.append(
                f"[Line {ctx.start.line}] Cannot apply negation to Neuron."
            )
        return self.visitChildren(ctx)

    # Check binary operations (e.g. PolyExp * PolyExp) #TODO
    def visitBinopExp(self, ctx):
        left_type = self.infer_type(ctx.getChild(0))
        right_type = self.infer_type(ctx.getChild(2))
        op = ctx.getChild(1).getText()
        if (left_type, right_type) in self.invalid_type_pairs.get(op, set()):
            self.errors.append(
                f"[Line {ctx.start.line}] Invalid type combination: {left_type} {op} {right_type}"
            )
        return self.visitChildren(ctx)

    def infer_type(self, ctx):
        text = ctx.getText()

        # Base Cases
        if any(
            text.startswith(prefix + "[") and any(k in text for k in ["L", "U"])
            for prefix in ["curr", "prev", "curr_list", "prev_0", "prev_1"]
        ):

            return "PolyExp"

        if any(
            text.startswith(prefix + "[") and any(k in text for k in ["l", "u"])
            for prefix in ["curr", "prev", "curr_list", "prev_0", "prev_1"]
        ):
            return "Float"

        if text in {"curr", "prev", "curr_list", "prev_0", "prev_1"}:
            return "Neuron"

        if text.isdigit():
            return "Int"

        if "." in text and text.replace(".", "", 1).isdigit():
            return "Float"

        # Binary expression
        if ctx.getChildCount() == 3:
            left_ctx = ctx.getChild(0)
            op = ctx.getChild(1).getText()
            right_ctx = ctx.getChild(2)

            left_type = self.infer_type(left_ctx)
            right_type = self.infer_type(right_ctx)

            # Neuron is not allowed in binary operations
            if "Neuron" in {left_type, right_type}:
                return "Neuron"

            # Int + Float -> Float
            if {left_type, right_type} == {"Int", "Float"}:
                return "Float"

            if left_type == right_type:
                return left_type

            # PolyExp dominates
            if "PolyExp" in {left_type, right_type}:
                return "PolyExp"

            # Float dominates Int
            if "Float" in {left_type, right_type}:
                return "Float"

            return "Unknown"

        if hasattr(ctx, "VAR"):  # have to put this in the end
            return self.var_types.get(text, "Unknown")

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
        return False, dsl, [f"[Parse Error] {str(e)}"]

    if error_listener.errors:
        return False, dsl, error_listener.errors

    checker = SemanticChecker()
    checker.visit(tree)

    if checker.errors:
        return False, dsl, checker.errors

    return True, dsl, []


if __name__ == "__main__":
    dsl = """
transformer deeppoly{
    HardSigmoid -> (max(prev[L],prev[U]),0,0,0);
}
    """
    result, dsl, errs = check_semantic(dsl)
    print(result)

    for e in errs:
        print(e)
