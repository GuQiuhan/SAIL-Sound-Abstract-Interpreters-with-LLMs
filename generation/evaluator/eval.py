import time
import traceback
from fractions import Fraction

import antlr4 as antlr
from z3 import *

from constraintflow.core import astBuilder
from constraintflow.core import astcf as AST
from constraintflow.core import astTC, astVisitor, dslLexer, dslParser
from constraintflow.provesound.lib.globals import *
from constraintflow.provesound.lib.optSolver import OptSolver
from constraintflow.provesound.lib.utils import *
from constraintflow.provesound.src.symbolicDNN import SymbolicDNN, populate_vars
from constraintflow.provesound.src.value import *

exptemp = None
op_ = None
case_ = 0


class Evaluate(astVisitor.ASTVisitor):
    # build upon the implementation of the `Verify` class.
    # support one op each time
    def __init__(self):
        self.shape = {}
        self.F = {}
        self.theta = {}
        self.Nprev = 3
        self.Nsym = 3
        self.number = Number()
        self.M = {}
        self.V = {}
        self.C = []
        self.E = []
        self.old_eps = []
        self.old_neurons = []
        self.store = {}
        self.arrayLens = {}
        self.solver = OptSolver()

    def visitShapeDecl(self, node):
        for (t, e) in node.elements.arglist:
            self.shape[e.name] = t.name

        self.constraint = node.p

    def visitFunc(self, node):
        self.F[node.decl.name.name] = node

    def visitTransformer(self, node):
        self.theta[node.name.name] = node

    def visitFlow(self, node):
        global exptemp
        global op_
        global case_
        shape_result = dict()
        node = self.theta[node.trans.name]
        for op_i in range(len(node.oplist.olist)):  # actually will only have one op
            self.E.clear()

            hasE = node.oplist.opsE[op_i]
            op = node.oplist.olist[op_i]
            store = self.store
            arrayLens = self.arrayLens
            prevLength = (Int("prevLength"), "Int")
            if self.Nprev > 1:
                self.C.append(prevLength[0] > 0)
                self.C.append(prevLength[0] <= self.Nprev)

            op_ = op.op.op_name
            case_ = 0

            # print(f"{op_} transformer")
            update_start_time()

            if (
                op_ == "Relu"
                or op_ == "Relu6"
                or op_ == "Abs"
                or op_ == "rev_Relu"
                or op_ == "rev_Relu6"
                or op_ == "rev_Abs"
                or op_ == "rev_Maxpool"
                or op_ == "HardTanh"
                or op_ == "rev_HardTanh"
                or op_ == "HardSigmoid"
                or op_ == "rev_HardSigmoid"
                or op_ == "HardSwish"
                or op_ == "rev_HardSwish"
            ):
                nprev = 1
            elif (
                op_ == "Neuron_mult"
                or op_ == "Neuron_add"
                or op_ == "Neuron_max"
                or op_ == "Neuron_min"
                or op_ == "rev_Neuron_mult"
                or op_ == "rev_Neuron_add"
                or op_ == "rev_Neuron_max"
                or op_ == "rev_Neuron_min"
            ):
                nprev = 2
            else:
                nprev = self.Nprev

            required_neurons = ["curr", "prev"]
            is_list = False
            if "Affine" in op_ or "pool" in op_:
                is_list = True

            if op_ == "rev_Maxpool":
                required_neurons = ["curr", "prev", "curr_list"]
            elif (
                op_ == "Neuron_mult"
                or op_ == "Neuron_add"
                or op_ == "Neuron_max"
                or op_ == "Neuron_min"
                or op_ == "rev_Neuron_mult"
                or op_ == "rev_Neuron_add"
                or op_ == "rev_Neuron_max"
                or op_ == "rev_Neuron_min"
                or op_ == "Neuron_list_mult"
            ):
                required_neurons = ["curr", "prev_0", "prev_1"]

            s = SymbolicDNN(
                self.store,
                self.F,
                self.constraint,
                self.shape,
                nprev,
                self.Nsym,
                self.number,
                self.M,
                self.V,
                self.C,
                self.E,
                self.old_eps,
                self.old_neurons,
                self.solver,
                self.arrayLens,
                prevLength,
            )
            s.ss.hasE = hasE

            if "curr" in required_neurons:
                curr = Vertex("Curr")
                self.V[curr.name] = curr
                populate_vars(
                    s.vars, curr, self.C, self.store, s.ss, self.constraint, self.number
                )
                store["curr"] = (curr.name, "Neuron")
            if "prev" in required_neurons:
                prev = []
                for i in range(nprev):
                    p = Vertex("Prev" + str(i))
                    prev.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                if is_list:
                    store["prev"] = prev
                else:
                    store["prev"] = prev[0]
                if len(prev) > 1:
                    arrayLens[str(prev)] = prevLength

            if "curr_list" in required_neurons:
                curr_list = []
                for i in range(self.Nprev):
                    p = Vertex("curr_list" + str(i))
                    curr_list.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                store["curr_list"] = curr_list
                arrayLens[str(curr_list)] = prevLength

            if op_ == "Neuron_list_mult":
                prev_0 = []
                prev_1 = []
                for i in range(nprev):
                    p = Vertex("Prev0_" + str(i))
                    prev_0.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )

                    p = Vertex("Prev1_" + str(i))
                    prev_1.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                store["prev_0"] = prev_0
                arrayLens[str(prev_0)] = prevLength
                store["prev_1"] = prev_1
                arrayLens[str(prev_1)] = prevLength
            elif "prev_1" in required_neurons:
                prev = []
                for i in range(nprev):
                    p = Vertex("Prev" + str(i))
                    prev.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                store["prev_0"] = prev[0]
                store["prev_1"] = prev[1]
            elif "prev_0" in required_neurons:
                prev = []
                for i in range(nprev):
                    p = Vertex("Prev" + str(i))
                    prev.append((p.name, "Neuron"))
                    self.V[p.name] = p
                    populate_vars(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                store["prev_0"] = prev[0]

            curr_prime = Vertex("curr_prime" + str(op_i))
            s.V[curr_prime.name] = curr_prime

            # Define relationship between curr and prev
            if op_ == "Affine":
                if not "weight" in curr.symmap.keys():
                    curr.symmap["weight"] = [
                        (
                            Real(
                                "weight_curr"
                                + str(op_i)
                                + "_"
                                + str(self.number.nextn())
                            ),
                            "Float",
                        )
                        for i in range(nprev)
                    ]
                    arrayLens[str(curr.symmap["weight"])] = prevLength
                if not "bias" in curr.symmap.keys():
                    curr.symmap["bias"] = (
                        Real("bias_curr" + str(self.number.nextn())),
                        "Float",
                    )
                exptemp = curr.symmap["bias"]
                for i in range(len(prev)):
                    exptemp = ADD(
                        exptemp,
                        IF(
                            LEQ(i + 1, prevLength),
                            (MULT(prev[i], curr.symmap["weight"][i])),
                            (0, "Int"),
                        ),
                    )

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Neuron_list_mult":
                exptemp = (0, "Float")
                for i in range(nprev):
                    exptemp = ADD(
                        exptemp,
                        IF(
                            LEQ(i + 1, prevLength),
                            MULT(prev_0[i], prev_1[i]),
                            (0, "Int"),
                        ),
                    )

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp
            elif op_ == "rev_Affine":
                if not "equations" in curr.symmap.keys():
                    curr.symmap["equations"] = [
                        (
                            Real(
                                "equations_"
                                + str(op_i)
                                + "_"
                                + str(self.number.nextn())
                            ),
                            "PolyExp",
                        )
                        for i in range(nprev)
                    ]
                    arrayLens[str(curr.symmap["equations"])] = prevLength
                exptemp = (True, "Bool")
                j = 0
                for t in curr.symmap["equations"]:
                    exptemp = AND(
                        exptemp,
                        IF(LEQ(j + 1, prevLength), EQQ(curr.name, t), (True, "Bool")),
                    )
                    j = j + 1

                s.currop = s.ss.convertToZ3(exptemp)
            elif op_ == "Relu":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])
                exptemp = IF(GEQ(exptemp, (0, "Float")), exptemp, (0, "Float"))

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Relu6":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])
                exptemp = IF(GEQ(exptemp, (0, "Float")), exptemp, (0, "Float"))
                exptemp = IF(LEQ(exptemp, (6, "Float")), exptemp, (6, "Float"))

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Abs":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])
                exptemp = IF(GEQ(exptemp, (0, "Float")), exptemp, NEG(exptemp))

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "HardTanh":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])
                exptemp = IF(
                    LEQ(exptemp, (-1, "Float")),
                    -1,
                    IF(GEQ(exptemp, (1, "Float")), 1, exptemp),
                )

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "HardSigmoid":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                exptemp = If(
                    If((exptemp + 1) / 2 > 1, 1, (exptemp + 1) / 2) < 0,
                    0,
                    If((exptemp + 1) / 2 > 1, 1, (exptemp + 1) / 2),
                )

                s.currop = curr.name == exptemp

            elif op_ == "HardSwish":
                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                exptemp = If(
                    exptemp <= -3,
                    0,
                    If(exptemp >= 3, exptemp, exptemp * (exptemp + 3) / 6),
                )

                s.currop = curr.name == exptemp

            elif op_ == "Neuron_mult":
                exptemp = MULT(prev[0], prev[1])

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Neuron_add":
                exptemp = ADD(prev[0], prev[1])

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Neuron_max":
                exptemp = IF(GEQ(prev[0], prev[1]), prev[0], prev[1])

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Neuron_min":
                exptemp = IF(LEQ(prev[0], prev[1]), prev[0], prev[1])

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif (
                op_ == "rev_Neuron_mult"
            ):  # prev_0 is the output neuron and prev_1 is the other input neuron
                exptemp = EQQ(prev[0], MULT(curr.name, prev[1]))
                s.currop = s.ss.convertToZ3(exptemp)

            elif op_ == "rev_Neuron_add":
                exptemp = EQQ(prev[0], ADD(curr.name, prev[1]))
                s.currop = s.ss.convertToZ3(exptemp)

            elif op_ == "rev_Neuron_max":
                exptemp = EQQ(prev[0], IF(GEQ(curr.name, prev[1]), curr.name, prev[1]))
                s.currop = s.ss.convertToZ3(exptemp)

            elif op_ == "rev_Neuron_min":
                exptemp = EQQ(prev[0], IF(LEQ(curr.name, prev[1]), curr.name, prev[1]))
                s.currop = s.ss.convertToZ3(exptemp)

            elif op_ == "rev_Relu":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(GEQ(curr.name, (0, "Float")), curr.name, (0, "Float"))
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "rev_Relu6":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(GEQ(curr.name, (0, "Float")), curr.name, (0, "Float"))
                prevexp = IF(LEQ(curr.name, (6, "Float")), prevexp, (6, "Float"))
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "rev_Abs":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(GEQ(curr.name, (0, "Float")), curr.name, NEG(curr.name))
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "rev_HardTanh":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(GEQ(curr.name, (-1, "Float")), curr.name, (-1, "Float"))
                prevexp = IF(LEQ(curr.name, (1, "Float")), prevexp, (1, "Float"))
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "rev_HardSigmoid":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(
                    GEQ(curr.name, (-1, "Float")),
                    DIV(ADD(curr.name, (1, "Float")), (2, "Float")),
                    (0, "Float"),
                )
                prevexp = IF(LEQ(curr.name, (1, "Float")), prevexp, (1, "Float"))
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "rev_HardSwish":

                exptemp = (0, "Float")
                for i in range(len(prev)):
                    exptemp = ADD(exptemp, prev[i])

                temp_curr_name = ADD(curr.name, (3, "Float"))

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = IF(
                    GEQ(temp_curr_name, (0, "Float")), temp_curr_name, (0, "Float")
                )
                prevexp = IF(LEQ(temp_curr_name, (6, "Float")), prevexp, (6, "Float"))
                prevexp = DIV(prevexp, (6, "Float"))
                prevexp = MULT(prevexp, curr.name)
                s.currop = s.ss.convertToZ3(prevexp) == exptemp

            elif op_ == "Maxpool":
                exptemp = prev[0]
                for i in range(1, nprev):
                    cond = LEQ(i + 1, prevLength)
                    for j in range(i):
                        cond = AND(cond, GEQ(prev[i], prev[j]))
                    for j in range(i + 1, nprev):
                        cond = AND(
                            cond, OR(GEQ(prev[i], prev[j]), GT(j + 1, prevLength))
                        )

                    exptemp = IF(cond, prev[i], exptemp)

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Minpool":
                exptemp = prev[0]
                for i in range(1, nprev):
                    cond = LEQ(i + 1, prevLength)
                    for j in range(i):
                        cond = AND(cond, LEQ(prev[i], prev[j]))
                    for j in range(i + 1, nprev):
                        cond = AND(
                            cond, OR(LEQ(prev[i], prev[j]), GT(j + 1, prevLength))
                        )

                    exptemp = IF(cond, prev[i], exptemp)

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "Avgpool":
                exptemp = prev[0]
                summation = prev[0]
                for i in range(1, nprev):
                    summation = ADD(summation, prev[i])
                    exptemp = IF(
                        EQQ(i + 1, prevLength), DIV(summation, prevLength), exptemp
                    )

                exptemp = s.ss.convertToZ3(exptemp)
                s.currop = curr.name == exptemp

            elif op_ == "rev_Maxpool":
                total_list = curr_list + [(curr.name, "Neuron")]
                exptemp = total_list[0]
                for i in range(1, self.Nprev + 1):
                    cond = (True, "Bool")
                    for j in range(self.Nprev + 1):
                        if i != j:
                            cond = AND(cond, GEQ(total_list[i], total_list[j]))
                    exptemp = IF(cond, total_list[i], exptemp)
                prevexp = (0, "Float")
                for i in range(len(prev)):
                    prevexp = ADD(prevexp, prev[i])

                exptemp = s.ss.convertToZ3(exptemp)
                prevexp = s.ss.convertToZ3(prevexp)
                s.currop = prevexp == exptemp

            for m in curr.symmap.keys():
                curr_prime.symmap[m] = curr.symmap[m]

            computation = curr_prime.name == curr.name
            if len(s.ss.arrayLens) != 0:
                computation = [
                    s.currop,
                    computation,
                    prevLength[0] > 0,
                    prevLength[0] <= nprev,
                ] + s.ss.tempC
            else:
                computation = [s.currop, computation] + s.ss.tempC
            s.ss.tempC = []
            s.flag = True

            try:
                s.visit(op.ret)
            except:
                print(f"Induction failed for {op_}")
                reset_time()

                traceback.print_exc()
                continue
            # s.visit(op.ret)

            vallist = None
            update_generation_time()

            if isinstance(op.ret, AST.TransRetIfNode):
                vallist = self.visitTransRetIf(op.ret, s)
            else:
                vallist = self.visitTransRetBasic(op.ret, s)
            leftC = computation + s.ss.C

            # Evaluate and return
            try:
                results = self.evaluate(
                    leftC, vallist, s, curr_prime, computation, exptemp
                )
                print(f"Unsound Transformer Evaluation Result for {op_}:", results)
                return (op_, results)
            except:
                print(f"Evaluation failed for {op_}.")
                traceback.print_exc()
                continue

            reset_time()

        return (op_, None)  # exception

    def visitTransRetBasic(self, node, s):
        return s.ss.visit(node.exprlist)

    def visitTransRetIf(self, node, s):
        cond = s.ss.visit(node.cond)
        left = None
        right = None
        if isinstance(node.tret, AST.TransRetIfNode):
            left = self.visitTransRetIf(node.tret, s)
        else:
            left = self.visitTransRetBasic(node.tret, s)

        if isinstance(node.fret, AST.TransRetIfNode):
            right = self.visitTransRetIf(node.fret, s)
        else:
            right = self.visitTransRetBasic(node.fret, s)

        return IF(cond, left, right)

    def visitSeq(self, node: AST.SeqNode):
        r1 = self.visit(node.stmt1)
        r2 = self.visit(node.stmt2)
        return r2

    def visitProg(self, node):
        self.visit(node.shape)
        return self.visit(node.stmt)

    def evaluate(self, leftC, vallist, s, curr_prime, computation, exptemp):
        """
        Build upon `verify.py`
        Give evaluation when the transformer is unsound.
        #TODO: need to add the evaluation when the transformer is sound to improve precision

        Returns:
            List of List: [[[l,u,L,U], f(x)], [[l,u,L,U], f(x)]..]
            or
            None if it's sound
        """
        results = []
        global case_
        if isinstance(vallist, list):
            for (elem, val) in zip(self.shape.keys(), vallist):
                curr_prime.symmap[elem] = val

            conslist = [self.constraint]
            if isinstance(self.constraint, AST.ExprListNode):
                conslist = self.constraint.exprlist
            set_option(
                max_args=10000000,
                max_lines=1000000,
                max_depth=10000000,
                max_visited=1000000,
            )

            case_ += 1
            for cons_id, one_cons in enumerate(conslist):
                c = populate_vars(
                    s.vars,
                    curr_prime,
                    self.C,
                    self.store,
                    s.ss,
                    one_cons,
                    self.number,
                    False,
                )
                z3constraint = s.ss.convertToZ3(c)
                if isinstance(z3constraint, bool):
                    z3constraint = BoolSort().cast(z3constraint)
                if isinstance(exptemp, z3.z3.ArithRef) and not ("rev" in op_):
                    z3constraint = substitute(z3constraint, (curr_prime.name, exptemp))
                eps_constraints = []
                for eps in self.E:
                    eps_constraints.append(eps >= -1)
                    eps_constraints.append(eps <= 1)

                newLeftC = leftC + s.ss.tempC + eps_constraints + s.ss.C

                gen_time = time.time()
                update_generation_time()
                lhs = And(newLeftC)
                rhs = z3constraint
                w = self.solver.solve(lhs, rhs)
                end_time = time.time()
                update_verification_time()
                # if(not w):
                #   raise Exception(f"Constraint Unsound. Proved in {end_time - gen_time : .5f}s")

                # @qiuhan: catch counterexample
                if not w:
                    # Try to extract the counterexample from the last model
                    # Change the `OptSolver` class to save the last model
                    if (
                        hasattr(self.solver, "models")
                        and self.solver.models is not None
                    ):

                        # --- Start: Evaluate bounds using model ---
                        from z3 import Const, RealSort

                        for m in self.solver.models:
                            print(m)

                            def extract_model_subs(model):
                                subs = []
                                for d in model.decls():
                                    var_name = d.name()
                                    val = model[d]
                                    # print(var_name)
                                    try:
                                        val_str = str(val).replace("?", "")
                                        # Use Const(var_name, RealSort()) for substitution
                                        sort = val.sort()
                                        if sort == IntSort():
                                            value = float(val_str)
                                            z3_var = Int(var_name)
                                            z3_val = IntVal(value)
                                        elif sort == RealSort():
                                            z3_var = Real(var_name)
                                            # z3_val = RealVal(value)

                                            if is_rational_value(val):
                                                num = val.numerator_as_long()
                                                den = val.denominator_as_long()
                                                frac = Fraction(num, den)
                                                z3_val = RealVal(frac)
                                            else:
                                                value = float(val_str)
                                                z3_val = RealVal(value)

                                        else:
                                            value = float(val_str)
                                            # continue  # Skip Bool or uninterpreted sorts for now
                                            z3_var = Const(var_name, RealSort())
                                            z3_val = RealVal(value)

                                        subs.append((z3_var, z3_val))
                                        # subs.append((Const(var_name, RealSort()), RealVal(value)))
                                    except Exception:
                                        continue
                                return subs

                            subs = extract_model_subs(m)
                            # print(subs)

                            def evaluate_expr(expr):
                                if isinstance(expr, list):
                                    return [evaluate_expr(e) for e in expr]
                                elif isinstance(expr, (int, float)):
                                    return expr
                                elif isinstance(expr, IF):
                                    cond = s.ss.convertToZ3(expr.cond)
                                    cond_eval = simplify(substitute(cond, *subs))
                                    return evaluate_expr(
                                        expr.left if is_true(cond_eval) else expr.right
                                    )
                                elif isinstance(expr, tuple):
                                    z3expr = s.ss.convertToZ3(expr)
                                    if not is_expr(z3expr):
                                        if isinstance(z3expr, (int, float)):
                                            z3expr = RealVal(z3expr)
                                        else:
                                            raise Exception(
                                                f"Invalid Z3 expr: got {type(z3expr)} from {expr}"
                                            )
                                    return simplify(substitute(z3expr, *subs))
                                else:
                                    z3expr = s.ss.convertToZ3(expr)
                                    if not is_expr(z3expr):
                                        if isinstance(z3expr, (int, float)):
                                            z3expr = RealVal(z3expr)
                                        else:
                                            raise Exception(
                                                f"Unknown expr type after convert: {type(expr)} → {type(z3expr)}"
                                            )
                                    return simplify(substitute(z3expr, *subs))

                            try:
                                result = evaluate_expr(vallist)
                                precise = evaluate_expr(exptemp)

                                results.append([result, precise])

                            except Exception as e:
                                print(
                                    "⚠️ Evaluation error during counterexample processing:",
                                    str(e),
                                )
                                traceback.print_exc()
                            # --- End: Evaluate bounds using model ---

                        return results

                s.ss.tempC = []

        else:
            condz3 = s.ss.convertToZ3(vallist.cond)
            preC = leftC + s.ss.tempC
            s.ss.tempC = []
            results = self.evaluate(
                preC + [condz3], vallist.left, s, curr_prime, computation, exptemp
            )
            results = self.evaluate(
                preC + [Not(condz3)], vallist.right, s, curr_prime, computation, exptemp
            )

        return results


def eval(code: str, nprev=1, nsymb=1):
    lexer = dslLexer.dslLexer(antlr.InputStream(code))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)
    e = Evaluate()
    # e.Nprev = nprev
    # e.Nsym = nsymb
    op, results = e.visit(ast)
    return op, results  # need to sum up teh


def weight(op: str):
    """
    function-specific weight function
    """
    if op == "Abs":
        pass
    elif op == "Neuron_add":
        pass
    elif op == "Affine":
        pass
    elif op == "Avgpool":
        pass
    elif op == "HardSigmoid":
        pass
    elif op == "HardSwish":
        pass
    elif op == "HardTanh":
        pass
    elif op == "Neuron_max":
        pass
    elif op == "Maxpool":
        pass
    elif op == "Neuron_min":
        pass
    elif op == "Minpool":
        pass
    elif op == "Neuron_mult":
        pass
    elif op == "Relu":
        pass
    elif op == "Relu6":
        pass
    return 1


def ds(code: str, certifier: str):
    """
    D_s = Sum (w(x)*d(x))

    Args:
        code (str): dsl

    Return: float
    """

    op, results = eval(code)

    sum = 0

    if certifier == "deeppoly":
        for r in results:

            l = z3_val_to_float(r[0][0])
            u = z3_val_to_float(r[0][1])
            L = z3_val_to_float(r[0][2])
            U = z3_val_to_float(r[0][3])
            fx = z3_val_to_float(r[1])
            w = weight(op)
            sum += (
                max(0, l - fx) + max(0, fx - u) + max(0, L - fx) + max(0, fx - U)
            ) * w
    else:
        pass

    return sum


def z3_val_to_float(val):
    if isinstance(val, z3.RatNumRef):
        numerator = val.numerator_as_long()
        denominator = val.denominator_as_long()
        float_val = float(Fraction(numerator, denominator))
        return float_val


if __name__ == "__main__":

    dsl = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);

func replace_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[L]) : (coeff * n[U]);
func replace_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[U]) : (coeff * n[L]);

func priority(Neuron n) = n[layer];
func priority2(Neuron n) = -n[layer];

func stop(Int x, Neuron n, Float coeff) = true;

func backsubs_lower(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_lower){e <= n}).map(simplify_lower);
func backsubs_upper(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_upper){e >= n}).map(simplify_upper);

func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];

func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);

func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
func f2(Float x) = x * ((x + 3) / 6);
func f3(Neuron n) = max(f2(n[l]), f2(n[u]));

func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);

transformer deeppoly{
   Maxpool -> len(argmax(prev, f)) > 0 ? (max(prev[l]), max(prev[u]),  avg(argmax(prev, f)), avg(argmax(prev, f))) : (max(prev[l]), max(prev[u]), max(prev[l]), 0);

}

flow(forward, priority, true, deeppoly);
    """
    # ce_string1 = "Prev0_l_5 = 1\nPrev0_U_8 = 1\nPrev0 = 1\nCurr_u_2 = 0\nPrev0_u_6 = 1\nPrev0_L_7 = 1\nCurr_U_4 = 0\nCurr = 0\ncurr_prime0 = 0\nCurr_l_1 = 0\nCurr_L_3 = 0"
    # ce_string2 = "Prev0_l_5 = -5\nPrev0_U_8 = 1\nPrev0 = 1\nCurr_u_2 = 0\nPrev0_u_6 = -5\nPrev0_L_7 = 1\nCurr_U_4 = 0\nCurr = 0\ncurr_prime0 = 0\nCurr_l_1 = 0\nCurr_L_3 = 0"
    # ce_string3 = "V13_l_14 = 0\n  Curr_u_2 = 1\n  Prev0_u_6 = 0\n  V13_u_15 = 0\n  Curr_L_3 = 1\n  Prev0_L_7 = 0\n  curr_prime0 = 1\n  V13_L_16 = 0\n  Curr_U_4 = 1\n  out_trav_X19 = 1\n  Prev0_U_8 = 0\n  Curr_l_1 = 1\n  V13_U_17 = 0\n  Prev0_l_5 = 0\n  out_trav_X23 = 1\n  V13 = 0\n  out_trav_c24 = 0\n  out_trav_c20 = 0\n  Prev0 = 0\n  prevLength = 1\n  weight_curr0_9 = 0\n  bias_curr10 = 1\n  Curr = 1"

    certifier = "deeppoly"
    print(ds(dsl, certifier))


"""
 Affine -> (backsubs_lower(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), backsubs_upper(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), prev.dot(curr[weight]) + curr[bias], prev.dot(curr[weight]) + curr[bias]);

"""
