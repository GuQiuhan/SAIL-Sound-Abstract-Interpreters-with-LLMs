import logging
import re
from typing import Optional, Tuple

from generation.request import Client, TGIClient
from generation.utils import *
from generation.validator.semantics_check import check_semantic
from generation.validator.syntax_check import SyntaxChecker

# from generation import gen


MAX_RETRIES = 3


def make_block_extractor(certifier: str, cmpl: str):
    """
    Extract everything starting from the correct transformer keyword (deeppoly, ibp, deepz)
    until the closing brace '}' that balances the opening one.
    """

    keyword = certifier.lower()  # "deeppoly", "ibp", "deepz"

    match = re.search(rf"({re.escape(keyword)}\s*\{{)", cmpl)
    if not match:
        return ""

    start_idx = match.start()
    brace_count = 0
    for i in range(start_idx, len(cmpl)):
        if cmpl[i] == "{":
            brace_count += 1
        elif cmpl[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return "transformer " + cmpl[start_idx : i + 1].strip()

    return "transformer " + cmpl[start_idx:].strip()


prmpt_deeppoly = """
Here is the grammar of Constraintflow DSL:

'''
expr_list : expr COMMA expr_list
    |   expr ;

exprs: expr exprs
    | expr;

metadata: WEIGHT
    |   BIAS
    |   EQUATIONS
    |   LAYER ;

expr: FALSE                                         #false
    | TRUE                                          #true
    | IntConst                                      #int
    | FloatConst                                    #float
    | VAR                                           #varExp
    | EPSILON                                       #epsilon
    | CURR                                          #curr
    | PREV                                          #prev
    | PREV_0                                        #prev_0
    | PREV_1                                        #prev_1
    | CURRLIST                                      #curr_list
    | LPAREN expr RPAREN                            #parenExp
    | LSQR expr_list RSQR                           #exprarray
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr                               #binopExp
    | NOT expr                                      #not
    | MINUS expr                                    #neg
    | expr QUES expr COLON expr                     #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE expr RBRACE     #traverse
    | argmax_op LPAREN expr COMMA expr RPAREN       #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN                    #listOp
    | expr DOT MAP LPAREN expr RPAREN               #map
    | expr DOT MAPLIST LPAREN expr RPAREN           #map_list
    | expr DOT DOTT LPAREN expr RPAREN              #dot
    | expr DOT CONCAT LPAREN expr RPAREN            #concat
    | LP LPAREN lp_op COMMA expr COMMA expr RPAREN  #lp
    | VAR LPAREN expr_list RPAREN                   #funcCall
    | VAR exprs                                     #curry
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;
'''

DeepPoly certifier uses four kinds of bounds to approximate the operator: (Float l, Float u, PolyExp L, PolyExp U).
They must follow the constraints that: curr[l] <= curr <= curr[u] & curr[L] <= curr <= curr[U]. `curr` here means the current neuron, `prev` means the inputs to the operator.
When the operator takes multiple inputs, use `prev_0`, `prev_1`, ... to refer to each input.
So every transformer in each case of the case analysis must return four values. Use any funstions below if needed instead of use arithmetic operators.
Function you can use:
- func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
- func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);
- func replace_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[L]) : (coeff * n[U]);
- func replace_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[U]) : (coeff * n[L]);
- func priority(Neuron n) = n[layer];
- func priority2(Neuron n, Float c) = -n[layer];
- func stop(Neuron n) = false;
- func stop_traverse(Neuron n, Float c) = false;
- func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_upper){e >= n}).map(simplify_upper);
- func stop(Int x, Neuron n, Float coeff) = true;
- func backsubs_lower(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_upper){e >= n}).map(simplify_upper);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
- func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
- func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
- func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
- func f2(Float x) = x * ((x + 3) / 6);
- func f3(Neuron n) = max(f2(n[l]), f2(n[u]));
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func avg(List<Float> xs) = sum(xs) / len(xs);
- func argmax(List<Neuron> ns, (Neuron, Neuron -> Bool) cmp) = [ n | n in ns, forall m in ns. cmp(n, m) ];
- func argmin(List<Neuron> ns, (Neuron, Neuron -> Bool) cmp) = [ n | n in ns, forall m in ns. cmp(n, m) ];
- func sigma(Float x) = 1/(1+eps(-x));

"""
prmpt_deepz = """
Here is the grammar of Constraintflow DSL:

'''
expr_list : expr COMMA expr_list
    |   expr ;

exprs: expr exprs
    | expr;

metadata: WEIGHT
    |   BIAS
    |   EQUATIONS
    |   LAYER ;

expr: FALSE                                         #false
    | TRUE                                          #true
    | IntConst                                      #int
    | FloatConst                                    #float
    | VAR                                           #varExp
    | EPSILON                                       #epsilon
    | CURR                                          #curr
    | PREV                                          #prev
    | PREV_0                                        #prev_0
    | PREV_1                                        #prev_1
    | CURRLIST                                      #curr_list
    | LPAREN expr RPAREN                            #parenExp
    | LSQR expr_list RSQR                           #exprarray
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr                               #binopExp
    | NOT expr                                      #not
    | MINUS expr                                    #neg
    | expr QUES expr COLON expr                     #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE expr RBRACE     #traverse
    | argmax_op LPAREN expr COMMA expr RPAREN       #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN                    #listOp
    | expr DOT MAP LPAREN expr RPAREN               #map
    | expr DOT MAPLIST LPAREN expr RPAREN           #map_list
    | expr DOT DOTT LPAREN expr RPAREN              #dot
    | expr DOT CONCAT LPAREN expr RPAREN            #concat
    | LP LPAREN lp_op COMMA expr COMMA expr RPAREN  #lp
    | VAR LPAREN expr_list RPAREN                   #funcCall
    | VAR exprs                                     #curry
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;
'''

IBP certifier uses two kinds of bounds to overapproximate the operator: (Float l, Float u).
They must follow the constraints that: curr[l] <= curr <= curr[u]. `curr` here means the current neuron, `prev` means the inputs to the operator.
When the operator takes multiple inputs, use `prev_0`, `prev_1`, ... to refer to each input.
So every transformer in each case of the case analysis must return two values. Use any functions below if needed instead of using arithmetic operators.

Functions you can use:
- func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
- func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);
- func abs(Float x) = x > 0 ? x : -x;
- func max_lower(Neuron n1, Neuron n2) = n1[l]>=n2[l] ? n1[l] : n2[l];
- func max_upper(Neuron n1, Neuron n2) = n1[u]>=n2[u] ? n1[u] : n2[u];
- func min_lower(Neuron n1, Neuron n2) = n1[l]<=n2[l] ? n1[l] : n2[l];
- func min_upper(Neuron n1, Neuron n2) = n1[u]<=n2[u] ? n1[u] : n2[u];
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func sigma(Float x) = 1/(1+eps(-x));
- func priority(Neuron n) = n[layer];

"""
prmpt_ibp = """
Here is the grammar of Constraintflow DSL:

'''
expr_list : expr COMMA expr_list
    |   expr ;

exprs: expr exprs
    | expr;

metadata: WEIGHT
    |   BIAS
    |   EQUATIONS
    |   LAYER ;

expr: FALSE                                         #false
    | TRUE                                          #true
    | IntConst                                      #int
    | FloatConst                                    #float
    | VAR                                           #varExp
    | EPSILON                                       #epsilon
    | CURR                                          #curr
    | PREV                                          #prev
    | PREV_0                                        #prev_0
    | PREV_1                                        #prev_1
    | CURRLIST                                      #curr_list
    | LPAREN expr RPAREN                            #parenExp
    | LSQR expr_list RSQR                           #exprarray
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr                               #binopExp
    | NOT expr                                      #not
    | MINUS expr                                    #neg
    | expr QUES expr COLON expr                     #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE expr RBRACE     #traverse
    | argmax_op LPAREN expr COMMA expr RPAREN       #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN                    #listOp
    | expr DOT MAP LPAREN expr RPAREN               #map
    | expr DOT MAPLIST LPAREN expr RPAREN           #map_list
    | expr DOT DOTT LPAREN expr RPAREN              #dot
    | expr DOT CONCAT LPAREN expr RPAREN            #concat
    | LP LPAREN lp_op COMMA expr COMMA expr RPAREN  #lp
    | VAR LPAREN expr_list RPAREN                   #funcCall
    | VAR exprs                                     #curry
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;
'''

DeepZ certifier uses three components to overapproximate each operator: (Float l, Float u, SymExp z).
They must follow the constraints that: curr[l] <= curr <= curr[u] and curr In curr[z].
When the operator takes multiple inputs, use `prev_0`, `prev_1`, ... to refer to each input.
So every transformer in each case of the case analysis must return two values. Use any functions below if needed instead of using arithmetic operators.

Functions you can use:
- func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
- func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);
- func priority(Neuron n) = n[layer];
- func abs(Float x) = x > 0 ? x : -x;
- func s1(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
- func i1(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (s1(x1, x2) * x1);
- func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
- func f2(Float x) = x * ((x + 3) / 6);
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func sigma(Float x) = 1/(1+eps(-x));

"""


def model_repair(
    client: Client, is_chat: bool, certifier: str, dsl: str, err: str
) -> str:

    client = TGIClient(
        model="http://ggnds-serv-01.cs.illinois.edu:8086"
    )  # always use gpt5
    is_chat = True

    prmpt = ""
    if certifier == "deeppoly":
        prmpt = prmpt_deeppoly
    elif certifier == "deepz":
        prmpt = prmpt_deepz
    elif certifier == "ibp":
        prmpt = prmpt_ibp

    else:
        assert "Unknown Certifier"

    logging.info(f"\n💡 [Model Repair] Triggered model repair due to error:\n {err}")
    GlobalState.repair_rounds_now += 1
    print("\n💡 [Model Repair] Triggered model repair due to error:\n%s", err)
    prompt = f"""You are a DSL repair assistant. Fix the following DSL code based on the error.
[DSL GRAMMER]:
{prmpt}

[ERROR]:
{err}

[CODE]:
{dsl}

Return only the fixed DSL code.
"""
    completions = [
        client.chat(messages=[{"role": "user", "content": prompt}])
        if is_chat
        else client.textgen(prompt=prompt)
        for _ in range(3)
    ]
    # Return the first non-empty fix
    for code in completions:
        if code.strip():
            logging.info(f"\n💡 [Model Repair] Fix found. Fixed DSL:\n {code}")
            print("\n💡 [Model Repair] Fix found. Fixed DSL:\n", code)
            return code

    logging.info(
        f"\n⚠️ [Model Repair] No useful fix found, returning original DSL:\n {dsl}"
    )
    print("\n⚠️ [Model Repair] No useful fix found, returning original DSL:\n", dsl)

    return dsl  # fallback to original if nothing useful is returned


def check(
    certifier: str, client: Client, is_chat: bool, dsl: str
) -> Tuple[bool, str, Optional[str]]:
    """
    Check and repair syntactic and semantic errors in the dsl with both formal methods and llm tools.

    Return:
        (Bool: result, Str: dsl)

    """
    fixed_code = dsl

    # ---- Syntax Repair Phase ----
    syntax_attempt = 0
    syntax_checker = SyntaxChecker()
    syn_result = False
    syn_err = None
    while syntax_attempt < MAX_RETRIES:
        logging.info(f"[Syntax Phase] Attempt {syntax_attempt + 1}")
        syn_result, fixed_code, syn_err = syntax_checker.check(fixed_code)
        if syn_result:
            logging.info("[Syntax Phase] ✅ Syntax check passed.")
            break
        logging.info(f"[Syntax Phase] ❌ Syntax error:\n{syn_err}")
        fixed_code = model_repair(client, is_chat, certifier, fixed_code, syn_err)
        fixed_code = make_block_extractor(certifier, fixed_code)
        logging.info(f"[Syntax Phase] 🔧 Model-provided fix:\n{fixed_code}")
        syntax_attempt += 1

    if not syn_result:
        logging.error(
            f"[Syntax Phase] ❌ Failed after {MAX_RETRIES} attempts for code:\n {fixed_code}"
        )
        return False, fixed_code

    # ---- Semantic Repair Phase ----
    semantic_attempt = 0
    while semantic_attempt < MAX_RETRIES:
        logging.info(f"[Semantic Phase] Attempt {semantic_attempt + 1}")
        sem_result, _, sem_errs = check_semantic(fixed_code)
        if sem_result:
            logging.info(f"✅ All check passed for code:\n {fixed_code}")
            return True, fixed_code
        sem_err = "\n".join(sem_errs)
        logging.info(f"[Semantic Phase] ❌ Semantic error:\n{sem_err}")
        fixed_code = model_repair(client, is_chat, certifier, fixed_code, sem_err)
        fixed_code = make_block_extractor(certifier, fixed_code)
        logging.info(f"[Semantic Phase] 🔧 Model-provided fix:\n{fixed_code}")
        semantic_attempt += 1

    logging.error(
        f"[Semantic Phase] ❌ Failed after {MAX_RETRIES} attempts for code:\n",
        fixed_code,
    )
    return False, fixed_code


if __name__ == "__main__":
    client = TGIClient(model="http://ggnds-serv-01.cs.illinois.edu:8084")

    dsl = """
 transformer deepz{
    HardSwish -> (
        ((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[l]) : (((prev[l]) >= -3) ? f2(prev[l]) : 0)),
        ((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[u]) : (((prev[u]) <= 3) ? f2(prev[u]) : (prev[u]))),
        (
            (
                (((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[l]) : (((prev[l]) >= -3) ? f2(prev[l]) : 0))) +
                (((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[u]) : (((prev[u]) <= 3) ? f2(prev[u]) : (prev[u]))))
            ) / 2
        ) + (
            (
                (
                    ((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[u]) : (((prev[u]) <= 3) ? f2(prev[u]) : (prev[u])))
                ) - (
                    ((prev[u]) <= -3) ? 0 : (((prev[l]) >= 3) ? (prev[l]) : (((prev[l]) >= -3) ? f2(prev[l]) : 0))
                )
            ) / 2
        ) * eps
    );

    """

    success, fixed_code = check("deepz", client, True, dsl)

    if success:
        print("✅ DSL is valid.\n", fixed_code)
    else:
        print("❌ Invalid DSL even after fix:\n", fixed_code)
