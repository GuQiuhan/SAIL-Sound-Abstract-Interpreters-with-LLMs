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




func si(Float x) = sigma x;

transformer ibp{
  HardSigmoid -> (0, 1);
}



flow(forward, priority, true, ibp);
    """
    try:
        print(run_verifier_from_str(dsl))
    except:
        import traceback

        traceback.format_exc()


"""
func sig_deriv(Float z) = si(z) * (1 - si(z));
func lam(Float l, Float u)   = (si(u) - si(l)) / ((u - l) == 0 ? 1 : (u - l));
func lamp(Float l, Float u)  = min([sig_deriv(l), sig_deriv(u)]);
func sig_lower(Neuron x, Float l, Float u) =
    (l < 0) ? (si(l) + lamp(l,u) * (x[l] - l))
            : (si(l) + lam(l,u)  * (x[l] - l));

func sig_upper(Neuron x, Float l, Float u) =
    (u <= 0) ? (si(u) + lam(l,u)  * (x[u] - u))
             : (si(u) + lamp(l,u) * (x[u] - u));
"""
