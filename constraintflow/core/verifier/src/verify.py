from z3 import *

try:
    import dreal as dr

    HAS_DREAL = True
except ImportError:
    dreal = None
    HAS_DREAL = False

import time
import traceback

from constraintflow.core.ast_cflow import astcf as AST
from constraintflow.core.ast_cflow import astVisitor
from constraintflow.core.verifier.src.value import *

try:
    from constraintflow.core.verifier.lib.drealSolver import DRealSolver
    from constraintflow.core.verifier.src.symbolicDNN import SymbolicDNN, populate_vars
    from constraintflow.core.verifier.src.symbolicDNND import (
        SymbolicDNND,
        VertexD,
        populate_vars_dreal,
    )

    HAS_DREAL = True
except ImportError:
    HAS_DREAL = False

from constraintflow.core.verifier.lib.globals import *
from constraintflow.core.verifier.lib.optSolver import OptSolver
from constraintflow.core.verifier.lib.utils import *

exptemp = None
op_ = None
case_ = 0

NONLINEAR_OPS = {
    "Sigmoid",
    "Tanh",
    "Swish",
    "Exp",
    "Log",
    "Softplus",
    "Elu",
    "Selu",
    "Mish",
    "Softsign",
}

# This class implements the ProveSound algorithm. Given a ConstraintFlow
# program, it visits all the abstract transformers and checks their their soundness w.r.t
# a given soundness property (specified by the user as a part of the program).
# For each transformer, it initializes a SymbolicDNN object and then expands it using
# symbolicDNN.py. Then, using the soundness property, it generates the verification query
# using the symbolic semantics and checks its soundness using a z3 based solver implemented in
# src/optSolver.py. The verification time is stored in a dictionary and returned to the user.
class Verify(astVisitor.ASTVisitor):
    def __init__(self):
        # Initializes the Verify class with default parameters and data structures.
        # Attributes:
        #   shape (dict): A dictionary to store the abstract shape.
        #   F (dict): A dictionary to store function definitions.
        #   theta (dict): A dictionary to store transformer definitions.
        #   Nprev (int): Bound parameter, initialized to 3.
        #   Nsym (int): Bound parameter, initialized to 3.
        #   number (Number): An instance of the Number class to generate new variable names.
        #   M (dict): A dictionary to map the outputs of traverse and solver to values.
        #   V (dict): A dictionary to store symbolic variables and corresponding vertices.
        #   C (list): A list to store symbolic constraints.
        #   E (list): A list to store new symbolic variables.
        #   old_eps (list): A list to store existing symbolic epsilon values.
        #   old_neurons (list): A list to store existing symbolic neurons.
        #   store (dict): A dictionary to store variable values.
        #   arrayLens (dict): A dictionary to store lengths of arrays as z3 variables.
        #   solver (OptSolver): An instance of the OptSolver class.

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
        self.solver = OptSolver()  # z3 solver, for linear op
        if HAS_DREAL:
            self.drealsolver = DRealSolver()  # dreal solver, for non-linear op

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
        ret_dict = dict()
        node = self.theta[node.trans.name]
        for op_i in range(len(node.oplist.olist)):
            ret_dict[node.oplist.olist[op_i].op.op_name] = (0, 0)
            self.E.clear()

            hasE = node.oplist.opsE[op_i]
            op = node.oplist.olist[op_i]
            store = self.store
            arrayLens = self.arrayLens

            if op.op.op_name not in NONLINEAR_OPS:
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
                        s.vars,
                        curr,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
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
                            IF(
                                LEQ(j + 1, prevLength),
                                EQQ(curr.name, t),
                                (True, "Bool"),
                            ),
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
                        If((exptemp + 3) / 6 > 1, 1, (exptemp + 3) / 6) < 0,
                        0,
                        If((exptemp + 3) / 6 > 1, 1, (exptemp + 3) / 6),
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
                    exptemp = EQQ(
                        prev[0], IF(GEQ(curr.name, prev[1]), curr.name, prev[1])
                    )
                    s.currop = s.ss.convertToZ3(exptemp)

                elif op_ == "rev_Neuron_min":
                    exptemp = EQQ(
                        prev[0], IF(LEQ(curr.name, prev[1]), curr.name, prev[1])
                    )
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
                    prevexp = IF(
                        GEQ(curr.name, (0, "Float")), curr.name, NEG(curr.name)
                    )
                    s.currop = s.ss.convertToZ3(prevexp) == exptemp

                elif op_ == "rev_HardTanh":

                    exptemp = (0, "Float")
                    for i in range(len(prev)):
                        exptemp = ADD(exptemp, prev[i])

                    exptemp = s.ss.convertToZ3(exptemp)
                    prevexp = IF(
                        GEQ(curr.name, (-1, "Float")), curr.name, (-1, "Float")
                    )
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
                    prevexp = IF(
                        LEQ(temp_curr_name, (6, "Float")), prevexp, (6, "Float")
                    )
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
                    traceback.print_exc()
                    ret_dict[node.oplist.olist[op_i].op.op_name] = (
                        get_verification_time(),
                        get_generation_time(),
                    )
                    reset_time()
                    continue
                s.visit(op.ret)

                update_generation_time()
                vallist = None

                if isinstance(op.ret, AST.TransRetIfNode):
                    vallist = self.visitTransRetIf(op.ret, s)
                else:
                    vallist = self.visitTransRetBasic(op.ret, s)
                leftC = computation + s.ss.C

                # try:
                #   self.applyTrans(leftC, vallist, s, curr_prime, computation)
                #   print(f"Proved {op_}")
                # except:
                #   print(f"Transformer unsound for {op_}")

                # ret_dict[node.oplist.olist[op_i].op.op_name] = (get_verification_time(), get_generation_time())

                # @qiuhan
                counterex = None
                z3_model = None
                result = False
                try:
                    self.applyTrans(op_, leftC, vallist, s, curr_prime, computation)
                    result = True
                    print(f"Proved {op_}")
                except Exception as e:
                    print(f"Transformer unsound for {op_}")
                    result = False
                    # traceback.print_exc()
                    if len(e.args) > 1:
                        z3_model = e.args[1]
                        # if isinstance(z3_model, z3.ModelRef):
                        #    counterex = {}
                        #    for d in z3_model.decls():
                        #        counterex[d.name()] = z3_model[d]

                ret_dict[node.oplist.olist[op_i].op.op_name] = (
                    get_verification_time(),
                    get_generation_time(),
                    z3_model,  # @qiuhan: save the counterexample
                    result,
                )

                reset_time()
            else:  # for non-linear ops
                # 1)
                prevLength_dr = (dr.Variable("prevLength"), "Int")
                if self.Nprev > 1:
                    self.C.append(prevLength_dr[0] > 0)
                    self.C.append(prevLength_dr[0] <= self.Nprev)

                op_ = op.op.op_name
                case_ = 0
                update_start_time()

                # 2)
                # for sigmoid, tanh, swish
                nprev = 1
                required_neurons = ["curr", "prev"]
                is_list = False

                # 3)
                s = SymbolicDNND(
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
                    self.drealsolver,  # dReal
                    self.arrayLens,
                    prevLength_dr,
                )
                s.ss.hasE = hasE

                # 4) provide metadata
                if "curr" in required_neurons:
                    curr = VertexD("Curr")
                    self.V[curr.name] = curr
                    populate_vars_dreal(
                        s.vars,
                        curr,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                    store["curr"] = (curr.name, "Neuron")

                if "prev" in required_neurons:
                    p = VertexD("Prev0")
                    self.V[p.name] = p
                    populate_vars_dreal(
                        s.vars,
                        p,
                        self.C,
                        self.store,
                        s.ss,
                        self.constraint,
                        self.number,
                    )
                    store["prev"] = (p.name, "Neuron")

                    tmp = p.name
                    self.C.append(tmp >= -10)
                    self.C.append(tmp <= 10)

                curr_prime = VertexD(f"curr_prime{op_i}")
                s.V[curr_prime.name] = curr_prime

                # 5) op semantics
                if op_ in (
                    "Sigmoid",
                    "Tanh",
                    "Swish",
                    "Softsign",
                    "Softplus",
                    "Mish",
                    "Elu",
                    "Selu",
                ):
                    z_ir = store["prev"]
                    if isinstance(store["prev"], list):
                        z_ir = (0, "Float")
                        for i, vi in enumerate(store["prev"]):
                            z_ir = ADD(
                                z_ir, IF(LEQ(i + 1, prevLength_dr), vi, (0, "Int"))
                            )

                    z_dr = s.ss.convertToDReal(z_ir)

                    if op_ == "Sigmoid":

                        # sigmoid(z)=1 / (1 + exp(-z))
                        s.currop = curr.name == 1 / (1 + dr.exp(-z_dr))

                    elif op_ == "Tanh":
                        # tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
                        ez = dr.exp(z_dr)
                        emz = dr.exp(-z_dr)
                        s.currop = curr.name == (ez - emz) / (ez + emz)
                    """

                    elif op_ == "Swish":
                        # swish(z) = z * σ(z)
                        sigmoid_z = 1 / (1 + dr.exp(-z_dr))
                        s.currop = (curr.name == z_dr * sigmoid_z)

                    elif op_ == "Softsign":
                        # Softsign(z) = z / (1 + |z|)
                        s.currop = z_dr / (1 + dr.abs(z_dr))

                    elif op_ == "Softplus":
                        # Softplus(z) = log(1 + exp(z))
                        s.currop = dr.log(1 + dr.exp(z_dr))


                    elif op_ == "Mish":
                        # Mish(z) = z * tanh(softplus(z)) = z * tanh(log(1+exp(z)))
                        softplus_z = dr.log(1 + dr.exp(z_dr))
                        s.currop = z_dr * dr.tanh(softplus_z)

                    elif op_ == "Elu":
                        # ELU(z) = z if z > 0 else alpha * (exp(z) - 1)
                        alpha = 1.0
                        s.currop = dr.if_then_else(z_dr > 0, z_dr, alpha * (dr.exp(z_dr) - 1))

                    elif op_ == "Selu":
                        # SELU(z) = λ * z if z > 0 else λ * α * (exp(z) - 1)
                        # usually λ ≈ 1.0507, α ≈ 1.67326
                        alpha = 1.67326
                        lam = 1.0507
                        s.currop = dr.if_then_else(z_dr > 0, lam * z_dr, lam * alpha * (dr.exp(z_dr) - 1))

                    """

                else:
                    raise NotImplementedError(
                        f"dReal branch currently doesn't support {op_}"
                    )

                # 6) create curr'
                for m in curr.symmap.keys():
                    curr_prime.symmap[m] = curr.symmap[m]

                # 7) LHS

                computation = curr_prime.name == curr.name
                if len(s.ss.arrayLens) != 0:
                    computation = [
                        s.currop,
                        computation,
                        prevLength_dr[0] > 0,
                        prevLength_dr[0] <= nprev,
                    ] + getattr(s.ss, "tempC_dreal", [])
                else:
                    computation = [s.currop, computation] + getattr(
                        s.ss, "tempC_dreal", []
                    )
                s.ss.tempC_dreal = []
                s.flag = True

                # 8) vallist from the abstract transformer
                try:
                    s.visit(op.ret)
                except Exception:
                    print(f"Induction failed for {op_}")
                    traceback.print_exc()
                    ret_dict[node.oplist.olist[op_i].op.op_name] = (
                        get_verification_time(),
                        get_generation_time(),
                    )
                    reset_time()
                    continue

                if isinstance(op.ret, AST.TransRetIfNode):
                    vallist = self.visitTransRetIf(op.ret, s)
                else:
                    vallist = self.visitTransRetBasic(op.ret, s)

                # 9) LHS = computation + s.ss.C
                leftC = computation + s.ss.C

                # 10) LHS -> RHS δ-sat
                counterex = None
                result = False
                try:
                    self.applyTrans(op_, leftC, vallist, s, curr_prime, computation)
                    result = True
                    print(f"Proved {op_}")
                except Exception as e:
                    print(f"Transformer unsound for {op_}")
                    result = False
                    if len(e.args) > 1:
                        counterex = e.args[1]

                ret_dict[node.oplist.olist[op_i].op.op_name] = (
                    get_verification_time(),
                    get_generation_time(),
                    counterex,
                    result,
                )
                reset_time()
        return ret_dict

    def applyTrans(self, op_, leftC, vallist, s, curr_prime, computation):
        global case_

        is_nonlinear = op_ in NONLINEAR_OPS

        if not is_nonlinear:

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
                        z3constraint = substitute(
                            z3constraint, (curr_prime.name, exptemp)
                        )
                    eps_constraints = []
                    for eps in self.E:
                        eps_constraints.append(eps >= -1)
                        eps_constraints.append(eps <= 1)

                    newLeftC = leftC + s.ss.tempC + eps_constraints + s.ss.C

                    gen_time = time.time()
                    update_generation_time()
                    lhs = And(newLeftC)
                    rhs = z3constraint

                    # print(f'lhs: {lhs}\n')
                    # print(f'rhs: {rhs}')

                    w = self.solver.solve(lhs, rhs)
                    end_time = time.time()
                    update_verification_time()
                    # if(not w):
                    #   raise Exception(f"Constraint Unsound. Proved in {end_time - gen_time : .5f}s")

                    # @qiuhan: catch counterexample
                    model = None
                    if not w:
                        # Try to extract the counterexample from the last model
                        # Change the `OptSolver` class to save the last model
                        if (
                            hasattr(self.solver, "last_model")
                            and self.solver.last_model is not None
                        ):
                            model = self.solver.last_model
                        # traceback.print_exc()

                        raise Exception(
                            f"Constraint Unsound. Proved in {end_time - gen_time : .5f}s",
                            model,
                        )

                    s.ss.tempC = []

            else:
                condz3 = s.ss.convertToZ3(vallist.cond)
                preC = leftC + s.ss.tempC
                s.ss.tempC = []
                self.applyTrans(
                    op_, preC + [condz3], vallist.left, s, curr_prime, computation
                )
                self.applyTrans(
                    op_, preC + [Not(condz3)], vallist.right, s, curr_prime, computation
                )
        else:
            # dReal LHS -> RHS  = δ-unsat of LHS ∧ ¬rhs
            if isinstance(vallist, list):
                for (elem, val) in zip(self.shape.keys(), vallist):
                    curr_prime.symmap[elem] = val

                conslist = [self.constraint]
                if isinstance(self.constraint, AST.ExprListNode):
                    conslist = self.constraint.exprlist

                case_ += 1
                for cons_id, one_cons in enumerate(conslist):
                    # RHS: IR -> dReal
                    c_ir = populate_vars_dreal(
                        s.vars,
                        curr_prime,
                        self.C,
                        self.store,
                        s.ss,
                        one_cons,
                        self.number,
                        False,
                    )
                    rhs_dr = s.ss.convertToDReal(c_ir)

                    # LHS:
                    lhs_dr_list = []

                    # (a) computation(e.g. curr == sigmoid(z))
                    if isinstance(computation, list):
                        lhs_dr_list.extend(computation)
                    else:
                        lhs_dr_list.append(computation)

                    # (b) s.ss.C
                    if getattr(s.ss, "C", None):
                        for cc in s.ss.C:
                            lhs_dr_list.append(cc)

                    # (c) tempC_dreal
                    if getattr(s.ss, "tempC_dreal", None):
                        lhs_dr_list.extend(s.ss.tempC_dreal)

                    # (d) epsilon in [-1,1]
                    for eps in self.E:
                        eps_name = None
                        try:
                            eps_name = eps.decl().name()
                        except Exception:
                            eps_name = str(eps)
                        v_eps = dr.Variable(eps_name)
                        lhs_dr_list.append(v_eps >= -1)
                        lhs_dr_list.append(v_eps <= 1)

                    # print(lhs_dr_list)
                    # print(rhs_dr)
                    #  phi = (∧lhs) ∧ ¬rhs
                    if lhs_dr_list:
                        phi = dr.And(*(lhs_dr_list + [dr.Not(rhs_dr)]))
                    else:
                        phi = dr.Not(rhs_dr)

                    # dReal solver
                    gen_time = time.time()
                    update_generation_time()

                    is_sat, box, cex_list = self.drealsolver.solve(
                        phi,
                        lhs=[],
                        rhs=lambda _env: False,
                        delta=getattr(self.drealsolver, "delta", 1e-6),
                        random_try=16,
                        tol=1e-9,
                    )

                    end_time = time.time()
                    update_verification_time()

                    if is_sat:
                        # δ-sat -> counterexample (interval box）
                        first_cex = (
                            cex_list[0] if (cex_list and len(cex_list) > 0) else box
                        )
                        raise Exception(
                            f"Constraint Unsound (dReal). Proved in {end_time - gen_time:.5f}s",
                            first_cex,
                        )

                    # δ-unsat -> sound
                    s.ss.tempC_dreal = []

            else:
                # if statement
                cond_ir = vallist.cond
                preC_drl = []
                # then
                self.applyTrans(
                    op_,
                    preC_drl + [s.ss.convertToDReal(cond_ir)],
                    vallist.left,
                    s,
                    curr_prime,
                    computation,
                )
                # else
                self.applyTrans(
                    op_,
                    preC_drl + [s.ss.convertToDReal(NOT(cond_ir))],
                    vallist.right,
                    s,
                    curr_prime,
                    computation,
                )

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
