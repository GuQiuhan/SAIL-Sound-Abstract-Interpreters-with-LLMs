# miniastBuilder.py

from . import miniastcf as AST
from .miniDSLParser import miniDSLParser
from .miniDSLVisitor import miniDSLVisitor


class ASTBuilder(miniDSLVisitor):
    def visitTransformer(self, ctx: miniDSLParser.TransformerContext):
        op_list = self.visit(ctx.op_list())
        name = AST.VarNode(ctx.VAR().getText())
        # l = self.visit(ctx.trans_decl().expr_list())
        # expr_list = [v.name for v in l.exprlist]
        return AST.TransformerNode(name, op_list)

    # this O(n^2) to retain the order of the expressions
    def visitOp_list(self, ctx: miniDSLParser.Op_listContext):
        oplist = ctx.op_list()
        stmt = self.visit(ctx.op_stmt())

        if oplist:
            listNode = self.visit(oplist)
            newList = [stmt] + listNode.olist
            return AST.OpListNode(newList)
        else:
            return AST.OpListNode([stmt])

    def visitOp_stmt(self, ctx: miniDSLParser.Op_stmtContext):
        op = self.visit(ctx.operator())
        ret = self.visit(ctx.trans_ret())
        return AST.OpStmtNode(op, ret)

    def visitOperator(self, ctx: miniDSLParser.OperatorContext):
        return AST.OperatorNode(ctx.getText())

    def visitCondtrans(self, ctx: miniDSLParser.CondtransContext):
        cond = self.visit(ctx.expr())
        texpr = self.visit(ctx.trans_ret(0))
        fexpr = self.visit(ctx.trans_ret(1))
        return AST.TransRetIfNode(cond, texpr, fexpr)

    def visitParentrans(self, ctx: miniDSLParser.ParentransContext):
        return self.visit(ctx.trans_ret())

    def visitTrans(self, ctx: miniDSLParser.TransContext):
        expr_list = self.visit(ctx.expr_list())
        return AST.TransRetBasicNode(expr_list)

    def visitExpr_list(self, ctx: miniDSLParser.Expr_listContext):
        expr = self.visit(ctx.expr())
        if ctx.expr_list():
            exprs = ctx.expr_list()
            listNode = self.visit(exprs)
            newList = [expr] + listNode.exprlist
            return AST.ExprListNode(newList)
        else:
            return AST.ExprListNode([expr])

    def visitExpr(self, ctx: miniDSLParser.ExprContext):
        if ctx.getChildCount() == 3 and ctx.getChild(1).getText() in {
            "+",
            "-",
            "*",
            "/",
        }:
            left = self.visit(ctx.getChild(0))
            op = ctx.getChild(1).getText()
            right = self.visit(ctx.getChild(2))
            return AST.BinOpNode(left, op, right)
        elif ctx.VAR():
            return AST.VarNode(ctx.VAR().getText())
        elif ctx.IntConst():
            return AST.ConstIntNode(int(ctx.IntConst().getText()))
        elif ctx.FloatConst():
            return AST.ConstFloatNode(float(ctx.FloatConst().getText()))
        elif ctx.getChildCount() == 3 and ctx.getChild(0).getText() == "(":
            return self.visit(ctx.getChild(1))
        else:
            raise Exception(f"Unrecognized expr: {ctx.getText()}")
