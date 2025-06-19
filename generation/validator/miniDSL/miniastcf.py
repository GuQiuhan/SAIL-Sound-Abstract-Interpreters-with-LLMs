
class ASTNode:
    def __init__(self):
        self.node_name = "ASTNode"

class ExprNode(ASTNode):
    def __init__(self):
        super().__init__()
        self.node_name = "Expr"

class VarNode(ExprNode):
    def __init__(self, name):
        super().__init__()
        self.node_name = "Var"
        self.name = name

class ConstIntNode(ExprNode):
    def __init__(self, value):
        super().__init__()
        self.node_name = "Int"
        self.value = value

class ConstFloatNode(ExprNode):
    def __init__(self, value):
        super().__init__()
        self.node_name = "Float"
        self.value = value

class BinOpNode(ExprNode):
    def __init__(self, left, op, right):
        super().__init__()
        self.node_name = "BinOp"
        self.left = left
        self.op = op
        self.right = right

class TransRetNode(ASTNode):
    def __init__(self):
        super().__init__()
        self.node_name = "TransRet"

class TransRetBasicNode(TransRetNode):
    def __init__(self, exprlist):
        super().__init__()
        self.node_name = "TransRetBasic"
        self.exprlist = exprlist

class TransRetIfNode(TransRetNode):
    def __init__(self, cond, tret, fret):
        super().__init__()
        self.node_name = "TransRetIf"
        self.cond = cond
        self.tret = tret
        self.fret = fret

class ExprListNode(ASTNode):
    def __init__(self, exprlist):
        super().__init__()
        self.node_name = "ExprList"
        self.exprlist = exprlist

class OperatorNode(ASTNode):
    def __init__(self, op_name):
        super().__init__()
        self.node_name = "Operator"
        self.op_name = op_name

class OpStmtNode(ASTNode):
    def __init__(self, op, ret):
        super().__init__()
        self.node_name = "OpStmt"
        self.op = op
        self.ret = ret

class OpListNode(ASTNode):
    def __init__(self, olist):
        super().__init__()
        self.node_name = "OpList"
        self.olist = olist

class TransformerNode(ASTNode):
    def __init__(self, name, oplist):
        super().__init__()
        self.node_name = "Transformer"
        self.name = name
        self.oplist = oplist