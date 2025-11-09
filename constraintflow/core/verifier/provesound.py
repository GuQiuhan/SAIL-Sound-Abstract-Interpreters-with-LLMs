import antlr4 as antlr
import z3

from constraintflow.core.ast_cflow import astBuilder, astTC, dslLexer, dslParser
from constraintflow.core.verifier.src import verify


def provesound(program, nprev=1, nsymb=1):
    lexer = dslLexer.dslLexer(antlr.FileStream(program))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)
    v = verify.Verify()
    v.Nprev = nprev
    v.Nsym = nsymb
    v.visit(ast)


# @qiuhan:
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

        re = False
        for op_name, result in ret_dict.items():
            if len(result) == 4:
                _n, v_, model, re = result
                if v_ < 1.0 and isinstance(model, z3.ModelRef):
                    ce.append(f"Counterexample unsound for {op_name}:")
                    for d in model.decls():
                        ce.append(f"  {d.name()} = {model[d]}")

        if ce:
            return re, "\n".join(ce)
        else:
            return re, None

    except Exception as e:
        print(e)

        return False, None


if __name__ == "__main__":

    dsl = """
def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

func priority(Neuron n) = n[layer];

transformer ibp{
    HardSwish -> ((prev[u]) <= -3) ? (0, 0) :
                 ((prev[l]) >= 3) ? ((prev[l]), (prev[u])) :
                 ((prev[l]) >= -3 and (prev[u]) <= 3) ? ((prev[l] <= -1.5 and prev[u] <= -1.5) ? ((prev[u] * (prev[u] + 3) / 6), (prev[l] * (prev[l] + 3) / 6)) : ((prev[l] >= -1.5 and prev[u] >= -1.5) ? ((prev[l] * (prev[l] + 3) / 6), (prev[u] * (prev[u] + 3) / 6)) : (-0.375, max(prev[l] * (prev[l] + 3) / 6, prev[u] * (prev[u] + 3) / 6)))) :
                 ((prev[l]) < -3 and (prev[u]) <= 3) ? ((prev[u] <= -1.5) ? ((prev[u] * (prev[u] + 3) / 6), 0) : (-0.375, max(0, prev[u] * (prev[u] + 3) / 6))) :
                 ((prev[l]) >= -3 and (prev[u]) > 3) ? ((prev[l] >= -1.5) ? ((prev[l] * (prev[l] + 3) / 6), (prev[u])) : (-0.375, (prev[u]))) :
                 (-0.375, (prev[u]));
}

flow(forward, priority, true, ibp);


    """
    dsl2 = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);

func replace_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[L]) : (coeff * n[U]);
func replace_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[U]) : (coeff * n[L]);

func stop(Neuron n) = false;
func stop_traverse(Neuron n, Float c) = false;
func priority2(Neuron n, Float c) = -n[layer];
func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_lower){e <= n}).map(simplify_lower);
func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_upper){e >= n}).map(simplify_upper);
func priority(Neuron n) = n[layer];

func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);

func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
func f2(Float x) = x * ((x + 3) / 6);
func f3(Neuron n) = max(f2(n[l]), f2(n[u]));

transformer deeppoly{
    HardTanh ->
        (
            (prev[l] >= 1)
                ? (1, 1, 1, 1)
                : (
                    (prev[u] <= -1)
                        ? (-1, -1, -1, -1)
                        : (
                            max(-1, prev[l]),
                            min(1, prev[u]),
                            ((prev - prev[l]) * (min(1, prev[u]) - max(-1, prev[l])) / (prev[u] - prev[l])) + max(-1, prev[l]) - ((prev - prev[l]) * (max(-1, prev[l])) / (prev[u] - prev[l])),
                            ((prev - prev[l]) * (min(1, prev[u]) - max(-1, prev[l])) / (prev[u] - prev[l])) + max(-1, prev[l])
                        )
                )
        );
}


flow(forward, priority, true, deeppoly);


    """

    dsl3 = """
def Shape as (Float l, Float u, SymExp z){[(curr[u]>=curr),(curr In curr[z]),(curr[l]<=curr)]};
func priority(Neuron n) = n[layer];
transformer deepz{
   HardSigmoid -> (prev[u] <= -3) ? (0, 0, 0) :
                   ((prev[l] >= 3) ? (1, 1, 1) :
                   ((prev[l] >= -3) ?
                        ((prev[u] <= 3) ?
                            (((prev[l] + 3) / 6), ((prev[u] + 3) / 6), ((prev[z] / 6) + 0.5)) :
                            (((prev[l] + 3) / 6), 1, (((1 + ((prev[l] + 3) / 6)) / 2) + ((((1 - ((prev[l] + 3) / 6)) / 2)) * eps)))) :
                        ((prev[u] <= 3) ?
                            (0, ((prev[u] + 3) / 6), ((((prev[u] + 3) / 12)) + ((((prev[u] + 3) / 12)) * eps))) :
                            (0, 1, (0.5 + (0.5 * eps))))));
}

flow(forward, priority, true, deepz);
"""
    try:
        print(run_verifier_from_str(dsl3))
    except:
        import traceback

        traceback.format_exc()


"""
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);

func replace_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[L]) : (coeff * n[U]);
func replace_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[U]) : (coeff * n[L]);

func stop(Neuron n) = false;
func stop_traverse(Neuron n, Float c) = false;
func priority2(Neuron n, Float c) = -n[layer];
func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_lower){e <= n}).map(simplify_lower);
func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_upper){e >= n}).map(simplify_upper);
func priority(Neuron n) = n[layer];

func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);

func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
func f2(Float x) = x * ((x + 3) / 6);
func f3(Neuron n) = max(f2(n[l]), f2(n[u]));

transformer deeppoly{
    HardSwish -> ((prev[u]) <= -3) ? (0, 0, 0, 0)
    : (((prev[l]) >= 3) ? ((prev[l]), (prev[u]), (prev), (prev))
    : (((prev[l]) >= -3) ? (((prev[u]) <= 3) ? (
        (0 - max(0 - f2(prev[l]), max(0 - f2(prev[u]), 0.375))),
        (f3(prev)),
        (0 - max(0 - f2(prev[l]), max(0 - f2(prev[u]), 0.375))),
        ((slope(prev[l], prev[u]) * prev) + intercept(prev[l], prev[u]))
    ) : (
        (0 - max(0 - ((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])) , max(0 - ((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])) , 0.375))),
        (max(((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])), max(((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])), 0))),
        (0 - max(0 - ((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])) , max(0 - ((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])) , 0.375))),
        (max(((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])), max(((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])), 0)))
    )) : (
        (0 - max(0 - ((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])) , max(0 - ((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])) , 0.375))),
        (max(((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])), max(((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])), 0))),
        (0 - max(0 - ((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])) , max(0 - ((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])) , 0.375))),
        (max(((prev[l] <= -3) ? 0 : ((prev[l] < 3) ? f2(prev[l]) : prev[l])), max(((prev[u] <= -3) ? 0 : ((prev[u] < 3) ? f2(prev[u]) : prev[u])), 0)))
    )));
}
flow(forward, priority, true, deeppoly);


    """
