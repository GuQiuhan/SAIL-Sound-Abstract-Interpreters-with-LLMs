# Generated from miniDSL.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .miniDSLParser import miniDSLParser
else:
    from miniDSLParser import miniDSLParser

# This class defines a complete generic visitor for a parse tree produced by miniDSLParser.

class miniDSLVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by miniDSLParser#transformer.
    def visitTransformer(self, ctx:miniDSLParser.TransformerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#flowstmt.
    def visitFlowstmt(self, ctx:miniDSLParser.FlowstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#funcstmt.
    def visitFuncstmt(self, ctx:miniDSLParser.FuncstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#seqstmt.
    def visitSeqstmt(self, ctx:miniDSLParser.SeqstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#transstmt.
    def visitTransstmt(self, ctx:miniDSLParser.TransstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#func_decl.
    def visitFunc_decl(self, ctx:miniDSLParser.Func_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#op_list.
    def visitOp_list(self, ctx:miniDSLParser.Op_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#op_stmt.
    def visitOp_stmt(self, ctx:miniDSLParser.Op_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#trans_decl.
    def visitTrans_decl(self, ctx:miniDSLParser.Trans_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#operator.
    def visitOperator(self, ctx:miniDSLParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#condtrans.
    def visitCondtrans(self, ctx:miniDSLParser.CondtransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#parentrans.
    def visitParentrans(self, ctx:miniDSLParser.ParentransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#trans.
    def visitTrans(self, ctx:miniDSLParser.TransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#types.
    def visitTypes(self, ctx:miniDSLParser.TypesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#arglist.
    def visitArglist(self, ctx:miniDSLParser.ArglistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#expr_list.
    def visitExpr_list(self, ctx:miniDSLParser.Expr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#exprs.
    def visitExprs(self, ctx:miniDSLParser.ExprsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#lp.
    def visitLp(self, ctx:miniDSLParser.LpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#argmaxOp.
    def visitArgmaxOp(self, ctx:miniDSLParser.ArgmaxOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#prev.
    def visitPrev(self, ctx:miniDSLParser.PrevContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#maxOp.
    def visitMaxOp(self, ctx:miniDSLParser.MaxOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#dot.
    def visitDot(self, ctx:miniDSLParser.DotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#map_list.
    def visitMap_list(self, ctx:miniDSLParser.Map_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#float.
    def visitFloat(self, ctx:miniDSLParser.FloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#cond.
    def visitCond(self, ctx:miniDSLParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#epsilon.
    def visitEpsilon(self, ctx:miniDSLParser.EpsilonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#varExp.
    def visitVarExp(self, ctx:miniDSLParser.VarExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#neg.
    def visitNeg(self, ctx:miniDSLParser.NegContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#not.
    def visitNot(self, ctx:miniDSLParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#listOp.
    def visitListOp(self, ctx:miniDSLParser.ListOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#curr_list.
    def visitCurr_list(self, ctx:miniDSLParser.Curr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#curr.
    def visitCurr(self, ctx:miniDSLParser.CurrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#maxOpList.
    def visitMaxOpList(self, ctx:miniDSLParser.MaxOpListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#map.
    def visitMap(self, ctx:miniDSLParser.MapContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#exprarray.
    def visitExprarray(self, ctx:miniDSLParser.ExprarrayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#getMetadata.
    def visitGetMetadata(self, ctx:miniDSLParser.GetMetadataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#false.
    def visitFalse(self, ctx:miniDSLParser.FalseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#concat.
    def visitConcat(self, ctx:miniDSLParser.ConcatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#curry.
    def visitCurry(self, ctx:miniDSLParser.CurryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#int.
    def visitInt(self, ctx:miniDSLParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#prev_0.
    def visitPrev_0(self, ctx:miniDSLParser.Prev_0Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#prev_1.
    def visitPrev_1(self, ctx:miniDSLParser.Prev_1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#traverse.
    def visitTraverse(self, ctx:miniDSLParser.TraverseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#binopExp.
    def visitBinopExp(self, ctx:miniDSLParser.BinopExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#getElement.
    def visitGetElement(self, ctx:miniDSLParser.GetElementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#true.
    def visitTrue(self, ctx:miniDSLParser.TrueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#parenExp.
    def visitParenExp(self, ctx:miniDSLParser.ParenExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#funcCall.
    def visitFuncCall(self, ctx:miniDSLParser.FuncCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#argmax_op.
    def visitArgmax_op(self, ctx:miniDSLParser.Argmax_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#lp_op.
    def visitLp_op(self, ctx:miniDSLParser.Lp_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#max_op.
    def visitMax_op(self, ctx:miniDSLParser.Max_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#list_op.
    def visitList_op(self, ctx:miniDSLParser.List_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#binop.
    def visitBinop(self, ctx:miniDSLParser.BinopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#metadata.
    def visitMetadata(self, ctx:miniDSLParser.MetadataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by miniDSLParser#direction.
    def visitDirection(self, ctx:miniDSLParser.DirectionContext):
        return self.visitChildren(ctx)



del miniDSLParser