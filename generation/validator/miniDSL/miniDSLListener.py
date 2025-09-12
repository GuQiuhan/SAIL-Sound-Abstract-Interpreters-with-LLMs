# Generated from miniDSL.g4 by ANTLR 4.13.2
from antlr4 import *

if "." in __name__:
    from .miniDSLParser import miniDSLParser
else:
    from miniDSLParser import miniDSLParser

# This class defines a complete listener for a parse tree produced by miniDSLParser.
class miniDSLListener(ParseTreeListener):

    # Enter a parse tree produced by miniDSLParser#transformer.
    def enterTransformer(self, ctx: miniDSLParser.TransformerContext):
        pass

    # Exit a parse tree produced by miniDSLParser#transformer.
    def exitTransformer(self, ctx: miniDSLParser.TransformerContext):
        pass

    # Enter a parse tree produced by miniDSLParser#flowstmt.
    def enterFlowstmt(self, ctx: miniDSLParser.FlowstmtContext):
        pass

    # Exit a parse tree produced by miniDSLParser#flowstmt.
    def exitFlowstmt(self, ctx: miniDSLParser.FlowstmtContext):
        pass

    # Enter a parse tree produced by miniDSLParser#funcstmt.
    def enterFuncstmt(self, ctx: miniDSLParser.FuncstmtContext):
        pass

    # Exit a parse tree produced by miniDSLParser#funcstmt.
    def exitFuncstmt(self, ctx: miniDSLParser.FuncstmtContext):
        pass

    # Enter a parse tree produced by miniDSLParser#seqstmt.
    def enterSeqstmt(self, ctx: miniDSLParser.SeqstmtContext):
        pass

    # Exit a parse tree produced by miniDSLParser#seqstmt.
    def exitSeqstmt(self, ctx: miniDSLParser.SeqstmtContext):
        pass

    # Enter a parse tree produced by miniDSLParser#transstmt.
    def enterTransstmt(self, ctx: miniDSLParser.TransstmtContext):
        pass

    # Exit a parse tree produced by miniDSLParser#transstmt.
    def exitTransstmt(self, ctx: miniDSLParser.TransstmtContext):
        pass

    # Enter a parse tree produced by miniDSLParser#func_decl.
    def enterFunc_decl(self, ctx: miniDSLParser.Func_declContext):
        pass

    # Exit a parse tree produced by miniDSLParser#func_decl.
    def exitFunc_decl(self, ctx: miniDSLParser.Func_declContext):
        pass

    # Enter a parse tree produced by miniDSLParser#op_list.
    def enterOp_list(self, ctx: miniDSLParser.Op_listContext):
        pass

    # Exit a parse tree produced by miniDSLParser#op_list.
    def exitOp_list(self, ctx: miniDSLParser.Op_listContext):
        pass

    # Enter a parse tree produced by miniDSLParser#op_stmt.
    def enterOp_stmt(self, ctx: miniDSLParser.Op_stmtContext):
        pass

    # Exit a parse tree produced by miniDSLParser#op_stmt.
    def exitOp_stmt(self, ctx: miniDSLParser.Op_stmtContext):
        pass

    # Enter a parse tree produced by miniDSLParser#trans_decl.
    def enterTrans_decl(self, ctx: miniDSLParser.Trans_declContext):
        pass

    # Exit a parse tree produced by miniDSLParser#trans_decl.
    def exitTrans_decl(self, ctx: miniDSLParser.Trans_declContext):
        pass

    # Enter a parse tree produced by miniDSLParser#operator.
    def enterOperator(self, ctx: miniDSLParser.OperatorContext):
        pass

    # Exit a parse tree produced by miniDSLParser#operator.
    def exitOperator(self, ctx: miniDSLParser.OperatorContext):
        pass

    # Enter a parse tree produced by miniDSLParser#condtrans.
    def enterCondtrans(self, ctx: miniDSLParser.CondtransContext):
        pass

    # Exit a parse tree produced by miniDSLParser#condtrans.
    def exitCondtrans(self, ctx: miniDSLParser.CondtransContext):
        pass

    # Enter a parse tree produced by miniDSLParser#parentrans.
    def enterParentrans(self, ctx: miniDSLParser.ParentransContext):
        pass

    # Exit a parse tree produced by miniDSLParser#parentrans.
    def exitParentrans(self, ctx: miniDSLParser.ParentransContext):
        pass

    # Enter a parse tree produced by miniDSLParser#trans.
    def enterTrans(self, ctx: miniDSLParser.TransContext):
        pass

    # Exit a parse tree produced by miniDSLParser#trans.
    def exitTrans(self, ctx: miniDSLParser.TransContext):
        pass

    # Enter a parse tree produced by miniDSLParser#types.
    def enterTypes(self, ctx: miniDSLParser.TypesContext):
        pass

    # Exit a parse tree produced by miniDSLParser#types.
    def exitTypes(self, ctx: miniDSLParser.TypesContext):
        pass

    # Enter a parse tree produced by miniDSLParser#arglist.
    def enterArglist(self, ctx: miniDSLParser.ArglistContext):
        pass

    # Exit a parse tree produced by miniDSLParser#arglist.
    def exitArglist(self, ctx: miniDSLParser.ArglistContext):
        pass

    # Enter a parse tree produced by miniDSLParser#expr_list.
    def enterExpr_list(self, ctx: miniDSLParser.Expr_listContext):
        pass

    # Exit a parse tree produced by miniDSLParser#expr_list.
    def exitExpr_list(self, ctx: miniDSLParser.Expr_listContext):
        pass

    # Enter a parse tree produced by miniDSLParser#exprs.
    def enterExprs(self, ctx: miniDSLParser.ExprsContext):
        pass

    # Exit a parse tree produced by miniDSLParser#exprs.
    def exitExprs(self, ctx: miniDSLParser.ExprsContext):
        pass

    # Enter a parse tree produced by miniDSLParser#lp.
    def enterLp(self, ctx: miniDSLParser.LpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#lp.
    def exitLp(self, ctx: miniDSLParser.LpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#argmaxOp.
    def enterArgmaxOp(self, ctx: miniDSLParser.ArgmaxOpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#argmaxOp.
    def exitArgmaxOp(self, ctx: miniDSLParser.ArgmaxOpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#prev.
    def enterPrev(self, ctx: miniDSLParser.PrevContext):
        pass

    # Exit a parse tree produced by miniDSLParser#prev.
    def exitPrev(self, ctx: miniDSLParser.PrevContext):
        pass

    # Enter a parse tree produced by miniDSLParser#maxOp.
    def enterMaxOp(self, ctx: miniDSLParser.MaxOpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#maxOp.
    def exitMaxOp(self, ctx: miniDSLParser.MaxOpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#dot.
    def enterDot(self, ctx: miniDSLParser.DotContext):
        pass

    # Exit a parse tree produced by miniDSLParser#dot.
    def exitDot(self, ctx: miniDSLParser.DotContext):
        pass

    # Enter a parse tree produced by miniDSLParser#map_list.
    def enterMap_list(self, ctx: miniDSLParser.Map_listContext):
        pass

    # Exit a parse tree produced by miniDSLParser#map_list.
    def exitMap_list(self, ctx: miniDSLParser.Map_listContext):
        pass

    # Enter a parse tree produced by miniDSLParser#float.
    def enterFloat(self, ctx: miniDSLParser.FloatContext):
        pass

    # Exit a parse tree produced by miniDSLParser#float.
    def exitFloat(self, ctx: miniDSLParser.FloatContext):
        pass

    # Enter a parse tree produced by miniDSLParser#cond.
    def enterCond(self, ctx: miniDSLParser.CondContext):
        pass

    # Exit a parse tree produced by miniDSLParser#cond.
    def exitCond(self, ctx: miniDSLParser.CondContext):
        pass

    # Enter a parse tree produced by miniDSLParser#epsilon.
    def enterEpsilon(self, ctx: miniDSLParser.EpsilonContext):
        pass

    # Exit a parse tree produced by miniDSLParser#epsilon.
    def exitEpsilon(self, ctx: miniDSLParser.EpsilonContext):
        pass

    # Enter a parse tree produced by miniDSLParser#varExp.
    def enterVarExp(self, ctx: miniDSLParser.VarExpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#varExp.
    def exitVarExp(self, ctx: miniDSLParser.VarExpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#neg.
    def enterNeg(self, ctx: miniDSLParser.NegContext):
        pass

    # Exit a parse tree produced by miniDSLParser#neg.
    def exitNeg(self, ctx: miniDSLParser.NegContext):
        pass

    # Enter a parse tree produced by miniDSLParser#not.
    def enterNot(self, ctx: miniDSLParser.NotContext):
        pass

    # Exit a parse tree produced by miniDSLParser#not.
    def exitNot(self, ctx: miniDSLParser.NotContext):
        pass

    # Enter a parse tree produced by miniDSLParser#listOp.
    def enterListOp(self, ctx: miniDSLParser.ListOpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#listOp.
    def exitListOp(self, ctx: miniDSLParser.ListOpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#curr_list.
    def enterCurr_list(self, ctx: miniDSLParser.Curr_listContext):
        pass

    # Exit a parse tree produced by miniDSLParser#curr_list.
    def exitCurr_list(self, ctx: miniDSLParser.Curr_listContext):
        pass

    # Enter a parse tree produced by miniDSLParser#curr.
    def enterCurr(self, ctx: miniDSLParser.CurrContext):
        pass

    # Exit a parse tree produced by miniDSLParser#curr.
    def exitCurr(self, ctx: miniDSLParser.CurrContext):
        pass

    # Enter a parse tree produced by miniDSLParser#maxOpList.
    def enterMaxOpList(self, ctx: miniDSLParser.MaxOpListContext):
        pass

    # Exit a parse tree produced by miniDSLParser#maxOpList.
    def exitMaxOpList(self, ctx: miniDSLParser.MaxOpListContext):
        pass

    # Enter a parse tree produced by miniDSLParser#map.
    def enterMap(self, ctx: miniDSLParser.MapContext):
        pass

    # Exit a parse tree produced by miniDSLParser#map.
    def exitMap(self, ctx: miniDSLParser.MapContext):
        pass

    # Enter a parse tree produced by miniDSLParser#exprarray.
    def enterExprarray(self, ctx: miniDSLParser.ExprarrayContext):
        pass

    # Exit a parse tree produced by miniDSLParser#exprarray.
    def exitExprarray(self, ctx: miniDSLParser.ExprarrayContext):
        pass

    # Enter a parse tree produced by miniDSLParser#getMetadata.
    def enterGetMetadata(self, ctx: miniDSLParser.GetMetadataContext):
        pass

    # Exit a parse tree produced by miniDSLParser#getMetadata.
    def exitGetMetadata(self, ctx: miniDSLParser.GetMetadataContext):
        pass

    # Enter a parse tree produced by miniDSLParser#false.
    def enterFalse(self, ctx: miniDSLParser.FalseContext):
        pass

    # Exit a parse tree produced by miniDSLParser#false.
    def exitFalse(self, ctx: miniDSLParser.FalseContext):
        pass

    # Enter a parse tree produced by miniDSLParser#concat.
    def enterConcat(self, ctx: miniDSLParser.ConcatContext):
        pass

    # Exit a parse tree produced by miniDSLParser#concat.
    def exitConcat(self, ctx: miniDSLParser.ConcatContext):
        pass

    # Enter a parse tree produced by miniDSLParser#curry.
    def enterCurry(self, ctx: miniDSLParser.CurryContext):
        pass

    # Exit a parse tree produced by miniDSLParser#curry.
    def exitCurry(self, ctx: miniDSLParser.CurryContext):
        pass

    # Enter a parse tree produced by miniDSLParser#int.
    def enterInt(self, ctx: miniDSLParser.IntContext):
        pass

    # Exit a parse tree produced by miniDSLParser#int.
    def exitInt(self, ctx: miniDSLParser.IntContext):
        pass

    # Enter a parse tree produced by miniDSLParser#prev_0.
    def enterPrev_0(self, ctx: miniDSLParser.Prev_0Context):
        pass

    # Exit a parse tree produced by miniDSLParser#prev_0.
    def exitPrev_0(self, ctx: miniDSLParser.Prev_0Context):
        pass

    # Enter a parse tree produced by miniDSLParser#prev_1.
    def enterPrev_1(self, ctx: miniDSLParser.Prev_1Context):
        pass

    # Exit a parse tree produced by miniDSLParser#prev_1.
    def exitPrev_1(self, ctx: miniDSLParser.Prev_1Context):
        pass

    # Enter a parse tree produced by miniDSLParser#traverse.
    def enterTraverse(self, ctx: miniDSLParser.TraverseContext):
        pass

    # Exit a parse tree produced by miniDSLParser#traverse.
    def exitTraverse(self, ctx: miniDSLParser.TraverseContext):
        pass

    # Enter a parse tree produced by miniDSLParser#binopExp.
    def enterBinopExp(self, ctx: miniDSLParser.BinopExpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#binopExp.
    def exitBinopExp(self, ctx: miniDSLParser.BinopExpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#getElement.
    def enterGetElement(self, ctx: miniDSLParser.GetElementContext):
        pass

    # Exit a parse tree produced by miniDSLParser#getElement.
    def exitGetElement(self, ctx: miniDSLParser.GetElementContext):
        pass

    # Enter a parse tree produced by miniDSLParser#true.
    def enterTrue(self, ctx: miniDSLParser.TrueContext):
        pass

    # Exit a parse tree produced by miniDSLParser#true.
    def exitTrue(self, ctx: miniDSLParser.TrueContext):
        pass

    # Enter a parse tree produced by miniDSLParser#parenExp.
    def enterParenExp(self, ctx: miniDSLParser.ParenExpContext):
        pass

    # Exit a parse tree produced by miniDSLParser#parenExp.
    def exitParenExp(self, ctx: miniDSLParser.ParenExpContext):
        pass

    # Enter a parse tree produced by miniDSLParser#funcCall.
    def enterFuncCall(self, ctx: miniDSLParser.FuncCallContext):
        pass

    # Exit a parse tree produced by miniDSLParser#funcCall.
    def exitFuncCall(self, ctx: miniDSLParser.FuncCallContext):
        pass

    # Enter a parse tree produced by miniDSLParser#argmax_op.
    def enterArgmax_op(self, ctx: miniDSLParser.Argmax_opContext):
        pass

    # Exit a parse tree produced by miniDSLParser#argmax_op.
    def exitArgmax_op(self, ctx: miniDSLParser.Argmax_opContext):
        pass

    # Enter a parse tree produced by miniDSLParser#lp_op.
    def enterLp_op(self, ctx: miniDSLParser.Lp_opContext):
        pass

    # Exit a parse tree produced by miniDSLParser#lp_op.
    def exitLp_op(self, ctx: miniDSLParser.Lp_opContext):
        pass

    # Enter a parse tree produced by miniDSLParser#max_op.
    def enterMax_op(self, ctx: miniDSLParser.Max_opContext):
        pass

    # Exit a parse tree produced by miniDSLParser#max_op.
    def exitMax_op(self, ctx: miniDSLParser.Max_opContext):
        pass

    # Enter a parse tree produced by miniDSLParser#list_op.
    def enterList_op(self, ctx: miniDSLParser.List_opContext):
        pass

    # Exit a parse tree produced by miniDSLParser#list_op.
    def exitList_op(self, ctx: miniDSLParser.List_opContext):
        pass

    # Enter a parse tree produced by miniDSLParser#binop.
    def enterBinop(self, ctx: miniDSLParser.BinopContext):
        pass

    # Exit a parse tree produced by miniDSLParser#binop.
    def exitBinop(self, ctx: miniDSLParser.BinopContext):
        pass

    # Enter a parse tree produced by miniDSLParser#metadata.
    def enterMetadata(self, ctx: miniDSLParser.MetadataContext):
        pass

    # Exit a parse tree produced by miniDSLParser#metadata.
    def exitMetadata(self, ctx: miniDSLParser.MetadataContext):
        pass

    # Enter a parse tree produced by miniDSLParser#direction.
    def enterDirection(self, ctx: miniDSLParser.DirectionContext):
        pass

    # Exit a parse tree produced by miniDSLParser#direction.
    def exitDirection(self, ctx: miniDSLParser.DirectionContext):
        pass


del miniDSLParser
