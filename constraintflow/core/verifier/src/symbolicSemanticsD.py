# dreal-only semantics
import dreal as dr

from constraintflow.core.ast_cflow import astcf as AST
from constraintflow.core.ast_cflow import astVisitor
from constraintflow.core.verifier.lib.utils import *
from constraintflow.core.verifier.src.symbolicSemantics import SymbolicSemantics
from constraintflow.core.verifier.src.value import *


class SymbolicSemanticsD(SymbolicSemantics):
    def __init__(
        self, store, F, M, V, C, E, old_eps, old_neurons, shape, Nprev, Nsym, arrayLens
    ):
        super().__init__(
            store, F, M, V, C, E, old_eps, old_neurons, shape, Nprev, Nsym, arrayLens
        )
        self.tempC_dreal = []  # temp_constraints for MAX/MIN/IN

    def _AND(self, *xs):
        xs2 = []
        for x in xs:
            if x is True:
                continue
            if x is False:
                return False
            xs2.append(x)
        if not xs2:
            return True
        if len(xs2) == 1:
            return xs2[0]
        return dr.And(*xs2)

    def _OR(self, *xs):
        xs2 = []
        for x in xs:
            if x is False:
                continue
            if x is True:
                return True
            xs2.append(x)
        if not xs2:
            return False
        if len(xs2) == 1:
            return xs2[0]
        return dr.Or(*xs2)

    # def visitFuncCall(self, node: AST.FuncCallNode, preeval=False):

    # args_sem = self.visit(node.arglist) if not preeval else node.arglist.exprlist
    # fname = node.name if isinstance(node.name, str) else node.name.name
    # fname_lower = str(fname).lower()

    # if fname_lower == "exp":
    #    if len(args_sem) != 1:
    #        raise Exception("exp() expects exactly 1 argument")
    #    x_dr = self.convertToDReal(args_sem[0])
    #    return (dr.exp(x_dr), "Float")

    # if fname_lower == "log":
    #    if not (1 <= len(args_sem) <= 2):
    #        raise Exception("log() expects 1 or 2 arguments")
    #    x_dr = self.convertToDReal(args_sem[0])
    #    if len(args_sem) == 1:
    #        return (dr.log(x_dr), "Float")
    #    b_dr = self.convertToDReal(args_sem[1])
    #    return (dr.log(x_dr) / dr.log(b_dr), "Float")

    # return super().visitFuncCall(node, preeval)

    def visitUnOp(self, node: AST.UnOpNode):
        expr = self.visit(node.expr)

        if node.op == "-":
            return self.get_unop(expr, NEG)
        elif node.op == "!":
            return self.get_unop(expr, NOT)
        elif node.op == "sigma":
            x_dr = self.convertToDReal(expr)
            y_dr = 1.0 / (1.0 + dr.exp(-x_dr))
            return (y_dr, "Float")
        else:
            assert False

    def visitEpsilon(self, node: AST.EpsilonNode):
        self.hasE = True
        eps = dr.Variable("Eps_" + str(self.number.nextn()))
        self.C.append(eps <= 1)
        self.C.append(eps >= -1)
        self.E.append(eps)
        return (eps, "Noise")

    def visitLp(self, node):
        # Or(Not(lhs), rhs)
        expr = self.visit(node.expr)
        constraints = self.visit(node.constraints)

        if str(constraints) in self.arrayLens:
            x = self.arrayLens[str(constraints)]
            for i in range(len(constraints)):
                constraints[i] = OR(constraints[i], LT(x, i + 1))

        out = dr.Variable("Lp_" + str(self.number.nextn()))
        lhs_dr = self.convertToDReal(constraints)  # list -> And(...)
        rhs_dr = None
        if node.op == "maximize":
            rhs_dr = out >= self.convertToDReal(expr)
        else:
            rhs_dr = out <= self.convertToDReal(expr)

        self.C.append(self._OR(dr.Not(lhs_dr), rhs_dr))
        return out

    def visitMaxOpList(self, node: AST.MaxOpListNode):
        elist = self.visit(node.expr)
        if not isinstance(elist, list):
            raise Exception("Type error: expect list for MaxOpList")

        if isinstance(elist[0], list):
            l = min([len(i) for i in elist])
            ret = []
            listlens = set()
            for listi in elist:
                if str(listi) in self.arrayLens:
                    listlens.add(self.arrayLens[str(listi)])

            for i in range(l):
                e = [j[i] for j in elist]
                ret.append(self.get_max(e, node.op))

            if len(listlens) == 1:
                self.arrayLens[str(ret)] = listlens.pop()
            elif len(listlens) > 1:
                newlen = (dr.Variable("newlen" + str(self.number.nextn())), "Int")
                for ln in listlens:
                    self.C.append(self.convertToDReal(LEQ(newlen, ln)))
                self.arrayLens[str(ret)] = newlen
            return ret

        ret = self.get_max(elist, node.op)
        return ret

    def get_binop(self, left, right, f):
        if isinstance(left, list):
            l = min(len(left), len(right))
            listout = [self.get_binop(left[i], right[i], f) for i in range(l)]
            xl = self.arrayLens.get(str(left))
            xr = self.arrayLens.get(str(right))
            if xl is None and xr is None:
                return listout
            elif xl is None:
                self.arrayLens[str(listout)] = xr
            elif xr is None:
                self.arrayLens[str(listout)] = xl
            else:

                self.arrayLens[str(listout)] = IF(LEQ(xr, xl), xr, xl)
            return listout

        if f == MULT:
            return create_mult(left, right)
        if f == ADD:
            return create_add(left, right)
        if f == SUB:
            return create_sub(left, right)
        if f == DIV:
            return create_div(left, right)
        return f(left, right)

    def convertToDReal(self, node):

        if isinstance(node, (bool, int, float)):
            return float(node) if not isinstance(node, bool) else bool(node)
        if isinstance(node, dr.Variable):
            return node
        if isinstance(node, tuple):
            val, ty = node
            if ty in ("Float", "Int"):

                if isinstance(val, (int, float, bool)):
                    return float(val) if ty != "Bool" else bool(val)

                if isinstance(val, (dr.Variable, dr.Expression)):
                    return val

                return val if isinstance(val, dr.Variable) else float(val)
            if ty == "Bool":
                return bool(val)
            if ty in ("Neuron", "PolyExp", "SymExp"):
                sym = node[0]
                return sym if isinstance(sym, dr.Variable) else dr.Variable(str(sym))

        if isinstance(node, list):
            parts = [self.convertToDReal(n) for n in node]
            return self._AND(*parts)

        if isinstance(node, ADD):
            return self.convertToDReal(node.left) + self.convertToDReal(node.right)
        if isinstance(node, SUB):
            return self.convertToDReal(node.left) - self.convertToDReal(node.right)
        if isinstance(node, MULT):
            return self.convertToDReal(node.left) * self.convertToDReal(node.right)
        if isinstance(node, DIV):
            return self.convertToDReal(node.left) / self.convertToDReal(node.right)
        if isinstance(node, NEG):
            return -(self.convertToDReal(node.left))

        if isinstance(node, NOT):
            l = self.convertToDReal(node.left)
            return (not l) if isinstance(l, bool) else dr.Not(l)

        if isinstance(node, AND):
            l = self.convertToDReal(node.left)
            r = self.convertToDReal(node.right)
            return self._AND(l, r)

        if isinstance(node, OR):
            l = self.convertToDReal(node.left)
            r = self.convertToDReal(node.right)
            return self._OR(l, r)

        if isinstance(node, LT):
            return self.convertToDReal(node.left) < self.convertToDReal(node.right)
        if isinstance(node, GT):
            return self.convertToDReal(node.left) > self.convertToDReal(node.right)
        if isinstance(node, LEQ):
            return self.convertToDReal(node.left) <= self.convertToDReal(node.right)
        if isinstance(node, GEQ):
            return self.convertToDReal(node.left) >= self.convertToDReal(node.right)
        if isinstance(node, EQQ):
            return self.convertToDReal(node.left) == self.convertToDReal(node.right)
        if isinstance(node, NEQ):
            return dr.Not(self.convertToDReal(EQQ(node.left, node.right)))

        if isinstance(node, IF):
            c = self.convertToDReal(node.cond)
            t = self.convertToDReal(node.left)
            e = self.convertToDReal(node.right)
            return dr.if_then_else(c, t, e)

        # x in [min_eps(e), max_eps(e)]
        if isinstance(node, IN):

            z = self.convertToZono(node.right)  # SymExpValue(const, coeffs)
            c = z.const
            tot = (0, "Float")
            for coeff in z.coeffs.values():
                abs_coeff = IF(GEQ(coeff, (0, "Float")), coeff, NEG(coeff))
                tot = create_add(tot, abs_coeff)
            lo = create_sub(c, tot)
            hi = create_add(c, tot)
            x = self.convertToDReal(node.left)
            lo = self.convertToDReal(lo)
            hi = self.convertToDReal(hi)
            return self._AND(x >= lo, x <= hi)

        # MAX/MIN
        if isinstance(node, MAX) or isinstance(node, MIN):
            es = [self.convertToDReal(e) for e in node.e]
            y = dr.Variable("new_" + str(self.number.nextn()))

            if str(node.e) in self.arrayLens:
                lenexp_dr = self.convertToDReal(self.arrayLens[str(node.e)])
                side = []
                pick = []
                for i, ei in enumerate(es):
                    if isinstance(node, MAX):
                        side.append(self._OR(y >= ei, (i + 1) > lenexp_dr))
                    else:
                        side.append(self._OR(y <= ei, (i + 1) > lenexp_dr))
                    pick.append(self._AND(y == ei, (i + 1) <= lenexp_dr))
            else:
                side = [(y >= ei) if isinstance(node, MAX) else (y <= ei) for ei in es]
                pick = [(y == ei) for ei in es]

            if pick:
                self.tempC_dreal.append(self._OR(*pick))
            if side:
                self.tempC_dreal.append(self._AND(*side))
            return y

        return node
