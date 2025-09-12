# -*- coding: utf-8 -*-

try:
    import dreal as dr

    HAS_DREAL = True
except ImportError:
    dreal = None
    HAS_DREAL = False

from constraintflow.core.ast_cflow import astcf as AST
from constraintflow.core.ast_cflow import astVisitor
from constraintflow.core.verifier.lib.drealSolver import DRealSolver
from constraintflow.core.verifier.lib.globals import *
from constraintflow.core.verifier.src.symbolicSemanticsD import SymbolicSemanticsD
from constraintflow.core.verifier.src.value import *


# dReal Vertex(int --> Real)
class VertexD:
    def __init__(self, name_str: str):
        self.symmap = {}
        self.name_str = str(name_str)
        self.name = dr.Variable(self.name_str)


# dReal populate_vars
def populate_vars_dreal(vars, v: VertexD, C, store, ss, constraint, number, flag=True):
    for var in vars.keys():
        if var not in v.symmap:
            vname = v.name_str
            ty = vars[var]
            # dReal has no Int/Bool -> dr.Variable
            v.symmap[var] = (dr.Variable(f"{vname}_{var}_{number.nextn()}"), ty)

    store["curr_new"] = (v.name, "Neuron")

    Ctemp = []
    cons_list = constraint if isinstance(constraint, list) else [constraint]
    for cons in cons_list:
        visited_ir = ss.visit(cons)  # IR
        if not flag:
            ss.flag = True
        Ctemp.append(ss.convertToDReal(visited_ir))  # dReal
        if not flag:
            ss.flag = False

    del store["curr_new"]
    if flag:
        C += Ctemp
    return Ctemp


class getVars(astVisitor.ASTVisitor):
    def __init__(self, constraint, shape):
        self.constraint = constraint
        self.shape = shape
        self.vars = {}

    def visitExprList(self, node: AST.ExprListNode):
        for e in node.exprlist:
            self.visit(e)

    def visitBinOp(self, node: AST.BinOpNode):
        self.visit(node.left)
        self.visit(node.right)

    def visitUnOp(self, node: AST.UnOpNode):
        self.visit(node.expr)

    def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
        self.visit(node.expr)
        self.visit(node.func)

    def visitMaxOpList(self, node: AST.MaxOpListNode):
        self.visit(node.expr)

    def visitMaxOp(self, node: AST.MaxOpNode):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitVar(self, node: AST.VarNode):
        pass

    def visitNeuron(self, node: AST.NeuronNode):
        pass

    def visitInt(self, node: AST.ConstIntNode):
        pass

    def visitFloat(self, node: AST.ConstFloatNode):
        pass

    def visitBool(self, node: AST.ConstBoolNode):
        pass

    def visitEpsilon(self, node: AST.EpsilonNode):
        pass

    def visitTernary(self, node: AST.TernaryNode):
        self.visit(node.cond)
        self.visit(node.texpr)
        self.visit(node.fexpr)

    def visitTraverse(self, node: AST.TraverseNode):
        self.visit(node.expr)
        self.visit(node.priority)
        self.visit(node.stop)
        self.visit(node.func)

    def visitListOp(self, node: AST.ListOpNode):
        self.visit(node.expr)

    def visitMap(self, node: AST.MapNode):
        self.visit(node.expr)

    def visitDot(self, node: AST.DotNode):
        self.visit(node.left)
        self.visit(node.right)

    def visitConcat(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visitFuncCall(self, node: AST.FuncCallNode):

        pass

    def visitSeq(self, node: AST.SeqNode):
        self.visit(node.stmt1)
        self.visit(node.stmt2)

    def visitFlow(self, node):
        pass

    def visitProg(self, node: AST.ProgramNode):
        self.visit(node.shape)
        self.visit(node.stmt)

    def visitGetMetadata(self, node: AST.GetMetadataNode):
        n = node.metadata.name
        if n == "weight":
            type = "Float list"
        elif n == "bias":
            type = "Float"
        else:
            type = "Int"
        self.vars[n] = type

    def visitGetElement(self, node):
        self.vars[node.elem.name] = self.shape[node.elem.name]


class expandSymbolicDNND(astVisitor.ASTVisitor):
    def __init__(self, os, vars, constraint, number, Nsym):
        self.ss = os
        self.vars = vars
        self.constraint = constraint
        self.number = number
        self.Nsym = Nsym

    def visitExprList(self, node: AST.ExprListNode):
        for e in node.exprlist:
            self.visit(e)

    def visitBinOp(self, node: AST.BinOpNode):
        self.visit(node.left)
        self.visit(node.right)

    def visitUnOp(self, node: AST.UnOpNode):
        self.visit(node.expr)

    def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
        self.visit(node.expr)
        self.visit(node.func)

    def visitMaxOp(self, node: AST.MaxOpNode):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitMaxOpList(self, node: AST.MaxOpListNode):
        self.visit(node.expr)

    def visitVar(self, node: AST.VarNode):
        pass

    def visitNeuron(self, node: AST.NeuronNode):
        pass

    def visitInt(self, node: AST.ConstIntNode):
        pass

    def visitFloat(self, node: AST.ConstFloatNode):
        pass

    def visitBool(self, node: AST.ConstBoolNode):
        pass

    def visitEpsilon(self, node: AST.EpsilonNode):
        pass

    def visitTernary(self, node: AST.TernaryNode):
        self.visit(node.cond)
        self.visit(node.texpr)
        self.visit(node.fexpr)

    def visitTraverse(self, node: AST.TraverseNode):
        self.visit(node.expr)
        self.visit(node.priority)
        self.visit(node.stop)
        self.visit(node.func)

    def visitListOp(self, node: AST.ListOpNode):
        self.visit(node.expr)

    def get_map(self, val, node):
        if isinstance(val, IF):
            self.get_map(val.left, node)
            self.get_map(val.right, node)
        else:
            if isinstance(val, ADD) or isinstance(val, SUB):
                self.get_map(val.left, node)
                self.get_map(val.right, node)
            elif isinstance(val, MULT):
                lhstype = self.ss.get_type(val.left)
                rhstype = self.ss.get_type(val.right)

                if isinstance(node.func, AST.VarNode):
                    elist = []
                    fname = node.func
                else:
                    elist = self.visit(node.func)
                    fname = node.func

                if lhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.left, val.right])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
                elif rhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.right, val.left])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
            elif isinstance(val, DIV):
                lhstype = self.ss.get_type(val.left)
                if isinstance(node.func, AST.VarNode):
                    elist = []
                    fname = node.func
                else:
                    elist = self.visit(node.func)
                    fname = node.func

                if lhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.left, DIV(1, val.right)])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
            else:
                if val[1] in ("Neuron", "Noise"):
                    if isinstance(node.func, AST.VarNode):
                        elist = []
                        fname = node.func
                    else:
                        elist = self.visit(node.func)
                        fname = node.func

                    elist = AST.ExprListNode(elist + [val, 1])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)

    def visitMap(self, node: AST.MapNode):
        self.visit(node.expr)
        val = self.ss.visit(node.expr)
        self.get_map(val, node)

    def visitDot(self, node: AST.DotNode):
        self.visit(node.left)
        self.visit(node.right)

    def visitConcat(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visitFuncCall(self, node: AST.FuncCallNode, preeval=False):
        if isinstance(node.name, str):
            func = self.ss.F[node.name]
        else:
            func = self.ss.F[node.name.name]

        newvars = []
        oldvalues = {}

        if not preeval:
            self.visit(node.arglist)
            elist = self.ss.visit(node.arglist)
        else:
            elist = node.arglist.exprlist

        for (exp, (t, arg)) in zip(elist, func.decl.arglist.arglist):
            if arg.name in self.ss.store.keys():
                oldvalues[arg.name] = self.ss.store[arg.name]
            else:
                newvars.append(arg.name)
            self.ss.store[arg.name] = exp

        self.visit(func.expr)

        for v in newvars:
            del self.ss.store[v]
        for ov in oldvalues.keys():
            self.ss.store[ov] = oldvalues[ov]

    def get_getMetadata(self, val, node):
        if isinstance(val, list):
            for v in val:
                self.get_getMetadata(v, node)
        elif isinstance(val, IF):
            self.get_getMetadata(val.left, node)
            self.get_getMetadata(val.right, node)
        else:
            name = node.metadata.name
            if name == "equations":
                newlist = []
                for eq in self.ss.V[val[0]].symmap[name]:
                    if isinstance(eq, tuple):
                        oldvar = eq[0]
                        newvar = (dr.Variable(f"X{self.number.nextn()}"), "Float")
                        for i in range(self.Nsym):
                            if len(self.ss.old_neurons) == i:
                                neuron = VertexD(f"V{self.number.nextn()}")
                                self.ss.old_neurons.append(neuron)
                                self.ss.V[neuron.name] = neuron
                                populate_vars_dreal(
                                    self.vars,
                                    neuron,
                                    self.ss.C,
                                    self.ss.store,
                                    self.ss,
                                    self.constraint,
                                    self.number,
                                )
                            neuron = self.ss.old_neurons[i]
                            const = (dr.Variable(f"c{self.number.nextn()}"), "Float")
                            newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))
                        newlist.append(newvar)
                        # 旧等式与新等式一致：以 dReal 的等式加入 C
                        self.ss.C.append(self.ss.convertToDReal(EQQ(oldvar, newvar)))
                if newlist:
                    self.ss.V[val[0]].symmap[name] = newlist

    def visitGetMetadata(self, node: AST.GetMetadataNode):
        n = self.ss.visit(node.expr)
        self.get_getMetadata(n, node)

    def get_getElement(self, val, node):
        if isinstance(val, list):
            for v in val:
                self.get_getElement(v, node)
        elif isinstance(val, IF):
            self.get_getElement(val.left, node)
            self.get_getElement(val.right, node)
        else:
            name = node.elem.name
            target = self.ss.V[val[0]].symmap[name]
            if isinstance(target, tuple) and target[1] == "PolyExp":
                oldvar = target[0]
                newvar = (dr.Variable(f"X{self.number.nextn()}"), "Float")
                for i in range(self.Nsym):
                    if len(self.ss.old_neurons) == i:
                        neuron = VertexD(f"V{self.number.nextn()}")
                        self.ss.old_neurons.append(neuron)
                        self.ss.V[neuron.name] = neuron
                        populate_vars_dreal(
                            self.vars,
                            neuron,
                            self.ss.C,
                            self.ss.store,
                            self.ss,
                            self.constraint,
                            self.number,
                        )
                    neuron = self.ss.old_neurons[i]
                    const = (dr.Variable(f"c{self.number.nextn()}"), "Float")
                    newvar = ADD(newvar, MULT(const, (neuron.name, "Neuron")))

                self.ss.V[val[0]].symmap[name] = newvar
                self.ss.C.append(self.ss.convertToDReal(EQQ(oldvar, newvar)))

            elif isinstance(target, tuple) and target[1] == "SymExp":
                oldvar = target[0]
                newvar = (dr.Variable(f"X{self.number.nextn()}"), "Float")
                for i in range(self.Nsym):
                    if len(self.ss.old_eps) == i:
                        epsilon = dr.Variable(f"eps_{self.number.nextn()}")
                        self.ss.old_eps.append(epsilon)
                        # eps in [-1,1]
                        self.ss.C.append(epsilon <= 1)
                        self.ss.C.append(epsilon >= -1)
                    epsilon = self.ss.old_eps[i]
                    const = (dr.Variable(f"X{self.number.nextn()}"), "Float")
                    newvar = ADD(newvar, MULT(const, (epsilon, "Noise")))

                self.ss.V[val[0]].symmap[name] = newvar
                self.ss.C.append(self.ss.convertToDReal(EQQ(oldvar, newvar)))
            return

    def visitGetElement(self, node: AST.GetElementNode):
        n = self.ss.visit(node.expr)
        self.get_getElement(n, node)


class SymbolicDNND(astVisitor.ASTVisitor):
    def __init__(
        self,
        store,
        F,
        constraint,
        shape,
        Nprev,
        Nsym,
        number,
        M,
        V,
        C,
        E,
        old_eps,
        old_neurons,
        dreal_solver: DRealSolver,
        arrayLens,
        prevLength,
    ):
        self.M = M
        self.V = V
        self.C = C
        self.E = E
        self.old_eps = old_eps
        self.old_neurons = old_neurons
        self.constraint = constraint
        self.store = store
        self.F = F
        self.shape = shape
        self.Nprev = Nprev
        self.Nsym = Nsym

        self.ss = SymbolicSemanticsD(
            self.store,
            self.F,
            self.M,
            self.V,
            self.C,
            self.E,
            self.old_eps,
            self.old_neurons,
            self.shape,
            self.Nprev,
            self.Nsym,
            arrayLens,
        )
        self.number = number
        self.currop = None
        g = getVars(self.constraint, self.shape)
        g.visit(self.constraint)
        self.vars = g.vars
        self.flag = False
        self.dreal_solver = dreal_solver
        self.prevLength = prevLength

    def visitInt(self, node):
        pass

    def visitEpsilon(self, node):
        pass

    def visitVar(self, node):
        pass

    def visitFloat(self, node):
        pass

    def visitBool(self, node):
        pass

    def visitBinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visitUnOp(self, node):
        self.visit(node.expr)

    def visitListOp(self, node):
        self.visit(node.expr)

    def visitDot(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visitConcat(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def get_getMetadata(self, n, node):
        if isinstance(n, IF):
            self.get_getMetadata(n.left, node)
            self.get_getMetadata(n.right, node)
        else:
            if not isinstance(n, list):
                if node.metadata.name not in self.V[n[0]].symmap:
                    newvar = None
                    if node.metadata.name == "bias":
                        newvar = (dr.Variable(f"X{self.number.nextn()}"), "Float")
                    elif node.metadata.name == "weight":
                        newvar_list = []
                        for _ in range(self.Nprev):
                            newvar_list.append(
                                (dr.Variable(f"X{self.number.nextn()}"), "Float")
                            )
                        newvar = newvar_list
                        self.ss.arrayLens[str(newvar_list)] = self.prevLength
                    elif node.metadata.name in ("layer", "serial", "local_serial"):
                        newvar = (dr.Variable(f"X{self.number.nextn()}"), "Int")
                    elif node.metadata.name == "equations":
                        newvar_list = []
                        for _ in range(self.Nprev):
                            newvar_list.append(
                                (dr.Variable(f"X{self.number.nextn()}"), "PolyExp")
                            )
                        newvar = newvar_list
                        self.ss.arrayLens[str(newvar_list)] = self.prevLength
                    self.V[n[0]].symmap[node.metadata.name] = newvar
            else:
                for ni in n:
                    self.get_getMetadata(ni, node)

    def visitGetMetadata(self, node):
        self.visit(node.expr)
        n = self.ss.visit(node.expr)
        self.get_getMetadata(n, node)

    def get_getElement(self, n, node):
        if isinstance(n, IF):
            self.get_getElement(n.left, node)
            self.get_getElement(n.right, node)
        else:
            if not isinstance(n, list):
                if node.elem.name not in self.V[n[0]].symmap:
                    ty = self.shape[node.elem.name]
                    newvar = (
                        dr.Variable(f"{node.elem.name}_X{self.number.nextn()}"),
                        ty,
                    )
                    self.V[n[0]].symmap[node.elem.name] = newvar
            else:
                for ni in n:
                    self.get_getElement(ni, node)

    def visitGetElement(self, node):
        self.visit(node.expr)
        n = self.ss.visit(node.expr)
        self.get_getElement(n, node)

    def visitGetElementAtIndex(self, node):
        self.visit(node.expr)

    def visitExprList(self, node):
        for e in node.exprlist:
            self.visit(e)

    def get_map(self, val, node):
        if isinstance(val, IF):
            self.get_map(val.left, node)
            self.get_map(val.right, node)
        else:
            if isinstance(val, ADD) or isinstance(val, SUB):
                self.get_map(val.left, node)
                self.get_map(val.right, node)
            elif isinstance(val, MULT):
                lhstype = self.ss.get_type(val.left)
                rhstype = self.ss.get_type(val.right)
                if isinstance(node.func, AST.VarNode):
                    elist = []
                    fname = node.func
                else:
                    elist = self.ss.visit(node.func)
                    fname = node.func

                if lhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.left, val.right])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
                elif rhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.right, val.left])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
            elif isinstance(val, DIV):
                lhstype = self.ss.get_type(val.left)
                if isinstance(node.func, AST.VarNode):
                    elist = []
                    fname = node.func
                else:
                    elist = self.ss.visit(node.func)
                    fname = node.func
                if lhstype in ("PolyExp", "SymExp", "Neuron", "Noise"):
                    elist = AST.ExprListNode(elist + [val.left, DIV(1, val.right)])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)
            else:
                if val[1] in ("Neuron", "Noise"):
                    if isinstance(node.func, AST.VarNode):
                        elist = []
                        fname = node.func
                    else:
                        elist = self.ss.visit(node.func)
                        fname = node.func
                    elist = AST.ExprListNode(elist + [val, 1])
                    fcall = AST.FuncCallNode(fname, elist)
                    self.visitFuncCall(fcall, True)

    def visitMap(self, node):
        self.visit(node.expr)
        expandSymbolicDNND(
            self.ss, self.vars, self.constraint, self.number, self.Nsym
        ).visit(node.expr)
        p = self.ss.visit(node.expr)
        self.get_map(p, node)

    def get_maplist(self, val, node):
        if isinstance(val, IF):
            self.get_maplist(val.left, node)
            self.get_maplist(val.right, node)
        else:
            for n in val:
                if isinstance(node.func, AST.VarNode):
                    elist = []
                else:
                    elist = self.ss.visit(node.func)
                elist = AST.ExprListNode(elist + [n])
                fcall = AST.FuncCallNode(node.func, elist)
                self.visitFuncCall(fcall, True)

    def visitMapList(self, node):
        self.visit(node.expr)
        p = self.ss.visit(node.expr)
        self.get_maplist(p, node)

    def visitFuncCall(self, node, preeval=False):
        name = node.name.name
        if isinstance(name, str):
            func = self.F[node.name.name]
        else:
            func = self.F[node.name.name.name]

        newvars = []
        oldvalues = {}

        if not preeval:
            self.visit(node.arglist)
            elist = self.ss.visit(node.arglist)
        else:
            elist = node.arglist.exprlist
        for (exp, (t, arg)) in zip(elist, func.decl.arglist.arglist):
            if arg.name in self.store.keys():
                oldvalues[arg.name] = self.store[arg.name]
            else:
                newvars.append(arg.name)
            self.store[arg.name] = exp

        self.visit(func.expr)

        for v in newvars:
            del self.store[v]
        for ov in oldvalues.keys():
            self.store[ov] = oldvalues[ov]

    def visitLp(self, node):
        self.visit(node.expr)
        self.visit(node.constraints)

    def visitArgmaxOp(self, node):
        self.visit(node.expr)

    def visitMaxOp(self, node):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitMaxOpList(self, node):
        self.visit(node.expr)

    def visitTernary(self, node):
        self.visit(node.cond)
        self.visit(node.texpr)
        self.visit(node.fexpr)

    # ------------------------------------------------------------------ #
    # dReal: (∧lhs) -> rhs  <=> delta-UNSAT of (∧lhs ∧ ¬rhs)
    # ------------------------------------------------------------------ #
    def _implication_holds_with_dreal(self, lhs_ctx_dr, rhs_dr) -> bool:
        if not lhs_ctx_dr:
            phi = dr.Not(rhs_dr)
        else:
            phi = dr.And(*(lhs_ctx_dr + [dr.Not(rhs_dr)]))

        is_sat, box, cexs = self.dreal_solver.solve(
            phi,
            lhs=[],  # lhs + rhs = phi
            rhs=lambda _env: False,
            delta=self.dreal_solver.delta,
            random_try=8,
            tol=1e-9,
        )
        return not is_sat  # delta-unsat => sound

    def _check_invariant(self, node):
        const = dr.Variable(f"inv_{self.number.nextn()}")
        input_ = (const, "Float")
        output_ = (const, "Float")

        for i in range(self.Nsym):
            if len(self.old_neurons) == i:
                neuron = VertexD(f"V{self.number.nextn()}")
                self.ss.V[neuron.name] = neuron
                populate_vars_dreal(
                    self.vars,
                    neuron,
                    self.C,
                    self.store,
                    self.ss,
                    self.constraint,
                    self.number,
                )
                self.old_neurons.append(neuron)

            neuron = self.old_neurons[i]
            coeff = (dr.Variable(f"inv_c{self.number.nextn()}"), "Float")

            elist_stop = (
                [] if isinstance(node.stop, AST.VarNode) else self.ss.visit(node.stop)
            )
            if isinstance(node.stop, AST.VarNode):
                self.visitFuncCall(
                    AST.FuncCallNode(
                        node.stop,
                        AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )
            elif isinstance(node.stop, AST.FuncCallNode):
                self.visitFuncCall(
                    AST.FuncCallNode(
                        node.stop.name,
                        AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )

            elist_func = (
                [] if isinstance(node.func, AST.VarNode) else self.ss.visit(node.func)
            )
            if isinstance(node.func, AST.VarNode):
                self.visitFuncCall(
                    AST.FuncCallNode(
                        node.func,
                        AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )

            if isinstance(node.stop, AST.VarNode):
                val_stop = self.ss.visitFuncCall(
                    AST.FuncCallNode(
                        node.stop,
                        AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )
            elif isinstance(node.stop, AST.FuncCallNode):
                val_stop = self.ss.visitFuncCall(
                    AST.FuncCallNode(
                        node.stop.name,
                        AST.ExprListNode(elist_stop + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )
            else:
                val_stop = self.ss.visit(node.stop)

            if isinstance(node.func, AST.VarNode):
                val_func = self.ss.visitFuncCall(
                    AST.FuncCallNode(
                        node.func,
                        AST.ExprListNode(elist_func + [(neuron.name, "Neuron"), coeff]),
                    ),
                    True,
                )
            else:
                val_func = self.ss.visit(node.func)

            input_ = ADD(input_, MULT(coeff, (neuron.name, "Neuron")))
            out_i = IF(val_stop, val_func, MULT(coeff, (neuron.name, "Neuron")))
            output_ = self.ss.get_binop(output_, out_i, ADD)

        old_val = self.ss.store[node.expr.name]
        self.ss.store[node.expr.name] = input_
        p_in_dr = self.ss.convertToDReal(self.ss.visit(node.p))

        self.ss.store[node.expr.name] = output_
        p_out_dr = self.ss.convertToDReal(self.ss.visit(node.p))

        lhs_ctx = []
        if getattr(self.ss, "C", None):
            for c in self.ss.C:
                lhs_ctx.append(self.ss.convertToDReal(c))
        if getattr(self.ss, "tempC_dreal", None):
            lhs_ctx.extend(self.ss.tempC_dreal)
        lhs_ctx.append(p_in_dr)

        update_generation_time()
        ok = self._implication_holds_with_dreal(lhs_ctx, p_out_dr)
        update_verification_time()
        if not ok:
            raise Exception("Induction step is not true")

        const2 = dr.Variable(f"out_trav_{self.number.nextn()}")
        output_poly = (const2, "Float")
        for i in range(self.Nsym):
            if len(self.ss.old_neurons) == i:
                neuron = VertexD(f"V{self.number.nextn()}")
                self.ss.V[neuron.name] = neuron
                populate_vars_dreal(
                    self.vars,
                    neuron,
                    self.C,
                    self.store,
                    self.ss,
                    self.constraint,
                    self.number,
                )
                self.ss.old_neurons.append(neuron)
            neuron = self.ss.old_neurons[i]
            coeff = (dr.Variable(f"out_trav_c{self.number.nextn()}"), "Float")
            output_poly = ADD(output_poly, MULT(coeff, (neuron.name, "Neuron")))

        self.ss.store[node.expr.name] = output_poly
        p_out_final_dr = self.ss.convertToDReal(self.ss.visit(node.p))
        self.ss.store[node.expr.name] = old_val
        self.ss.C.append(p_out_final_dr)

        return output_poly, p_out_final_dr

    def visitTraverse(self, node):
        self.visit(node.expr)
        e = self.ss.visit(node.expr)
        p_ir = self.ss.visit(node.p)
        p_dr = self.ss.convertToDReal(p_ir)

        # LHS (C ∧ tempC ∧ currop)
        lhs_ctx = []
        if getattr(self.ss, "C", None):
            for c in self.ss.C:
                lhs_ctx.append(self.ss.convertToDReal(c))
        if getattr(self.ss, "tempC_dreal", None):
            lhs_ctx.extend(self.ss.tempC_dreal)
        if self.currop is not None:
            lhs_ctx.append(self.ss.convertToDReal(self.currop))

        update_generation_time()
        ok = self._implication_holds_with_dreal(lhs_ctx, p_dr)
        update_verification_time()
        if not ok:
            raise Exception("Invariant is not true on input")

        output, prop_output = self._check_invariant(node)

        if isinstance(node.priority, AST.VarNode):
            p_name = node.priority.name
        else:
            p_name = self.ss.visit(node.priority)

        if isinstance(node.stop, AST.VarNode):
            s_name = node.stop.name
        else:
            s_name = self.ss.visit(node.stop)

        self.ss.M[TRAVERSE(e, node.direction, p_name, s_name, node.func.name)] = output

    def visitTransRetBasic(self, node):
        self.visit(node.exprlist)

    def visitTransRetIf(self, node):
        self.visit(node.cond)
        self.visit(node.tret)
        self.visit(node.fret)
