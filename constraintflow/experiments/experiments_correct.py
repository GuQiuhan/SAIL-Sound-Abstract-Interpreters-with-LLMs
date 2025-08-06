import sys

import antlr4 as antlr
from tabulate import tabulate
from z3 import *

from constraintflow.core import astBuilder, astTC, dslLexer, dslParser
from constraintflow.provesound.src import verify


def run_verifier(inputfile, nprev, nsymb):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)
    v = verify.Verify()
    v.Nprev = nprev
    v.Nsym = nsymb
    ret_dict = v.visit(ast)
    return ret_dict


def run_verifier_from_str(code: str, nprev=1, nsymb=1):
    """
    Verifies a DSL string (without writing to file) and returns result table.

    Args:
        code (str): The DSL program string.
        nprev (int): Number of prev neurons.
        nsymb (int): Number of symbolic inputs.

    Returns:
        (T/F, List[List[str or float]]: Formatted table with results (like main()).)
    """
    try:
        # Directly use InputStream from string
        lexer = dslLexer.dslLexer(antlr.InputStream(code))
        tokens = antlr.CommonTokenStream(lexer)
        parser = dslParser.dslParser(tokens)
        tree = parser.prog()
        ast = astBuilder.ASTBuilder().visit(tree)
        astTC.ASTTC().visit(ast)
        v = verify.Verify()
        v.Nprev = nprev
        v.Nsym = nsymb
        ret_dict = v.visit(ast)

        # Collect counterexamples
        ce = []
        model = None

        # print(ret_dict)
        for op_name, result in ret_dict.items():
            if len(result) == 3:
                _n, v_, model = result
                if v_ < 1.0 and isinstance(model, z3.ModelRef):
                    ce.append(f"Counterexample unsound for {op_name}:")
                    for d in model.decls():
                        ce.append(f"  {d.name()} = {model[d]}")

        if ce:
            return False, "\n".join(ce)
        else:
            return True, None

    except Exception as e:
        # Any parsing/type-check/verifier error
        print(str(e))

        return False, None


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
func slopeL(Float l, Float u) = (1 - l) / (2*(u - l));
func slopeU(Float l, Float u) = (u + 1) / (2*(u - l));

transformer deeppoly{
    HardTanh ->
        (prev[u] <= -1) ?
            (-1, -1, -1, -1) :
        (prev[l] >= 1) ?
            (1, 1, 1, 1) :
        ((prev[l] >= -1) and (prev[u] <= 1)) ?
            (prev[l], prev[u], prev, prev) :
        ((prev[l] < -1) and (prev[u] <= 1)) ?
            (max(-1, prev[l]), prev[u], max(-1, prev[l]), prev[u]) :
        ((prev[l] >= -1) and (prev[u] > 1)) ?
            (prev[l], min(1, prev[u]), prev[l], min(1, prev[u])) :
        (-1, 1, max(-1, prev[l]), min(1, prev[u]))
    ;
}

flow(forward, priority, true, deeppoly);
    """

    print(run_verifier_from_str(dsl))
"""
    certifier = sys.argv[1]
    nprev = int(sys.argv[2])
    nsym = int(sys.argv[3])
    ret_dict_correct = run_verifier(certifier, nprev, nsym)
    basicops = list(ret_dict_correct.keys())

    table = []
    row1 = []
    for b in basicops:
        row1.append(b)
        row1.append("")
    table.append(["Certifier"]+row1)
    heading = ['G', 'V']*len(basicops)
    table.append([" "]+heading)
    for c in [certifier]:
        row = [c]
        for b in basicops:
            row += [round(ret_dict_correct[b][1], 3), round(ret_dict_correct[b][0], 3)]
        table.append(row)
    print()
    print()
    print(tabulate(table))


"""
