"""
Please change the configuration accordingly before using this tool.
"""


# set models to use
MODEL_PORT_PAIRS = [
    {"model": "meta-llama/Llama-3.3-70B-Instruct", "port": 8081, "type": "hf"},
    {"model": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "port": 8080, "type": "hf"},
    {
        "model": "/share/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
        "port": 8082,
        "type": "vllm",
    },
    {"model": "gpt-4.1", "port": 8083, "type": "hf"},
    {"model": "gpt-4o", "port": 8084, "type": "hf"},
    {"model": "o4-mini", "port": 8085, "type": "hf"},
]

# for gen.py
PORT_MAP = {
    "deepseek": "http://ggnds-serv-01.cs.illinois.edu:8082",
    "llama-3.3": "http://ggnds-serv-01.cs.illinois.edu:8081",
    "llama-4": "http://ggnds-serv-01.cs.illinois.edu:8080",
    "gpt-4.1": "http://ggnds-serv-01.cs.illinois.edu:8083",
    "gpt-4o": "http://ggnds-serv-01.cs.illinois.edu:8084",
    "o4-mini": "http://ggnds-serv-01.cs.illinois.edu:8085",
}

# set max retry time every turn
MAX_RETRIES = 10


# prompting configuration

DEEPPOLY_CONSTRAINTFLOW = """
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepPoly transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

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
- func priority2(Neuron n) = -n[layer];
- func stop(Int x, Neuron n, Float coeff) = true;
- func backsubs_lower(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n, Int x) = (e.traverse(backward, priority2, stop(x), replace_upper){e >= n}).map(simplify_upper);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
- func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
- func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);
- func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
- func f2(Float x) = x * ((x + 3) / 6);
- func f3(Neuron n) = max(f2(n[l]), f2(n[u]));
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);

Don't add comments to DSL.
"""

prmpt_relu_deeppoly = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Relu -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev), (prev)) : (((prev[u]) <= 0) ? (0, 0, 0, 0) : (0, (prev[u]), 0, (((prev[u]) / ((prev[u]) - (prev[l]))) * (prev)) - (((prev[u]) * (prev[l])) / ((prev[u]) - (prev[l]))) ));
}
"""

prmpt_abs_deeppoly = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Abs -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev), (prev)) : (((prev[u]) <= 0) ? (0-(prev[u]), 0-(prev[l]), 0-(prev), 0-(prev)) : (0, max(prev[u], 0-prev[l]), prev, prev*(prev[u]+prev[l])/(prev[u]-prev[l]) - (((2*prev[u])*prev[l])/(prev[u]-prev[l]))) );
}
"""

prmpt_affine_deeppoly = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Affine -> (backsubs_lower(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), backsubs_upper(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), prev.dot(curr[weight]) + curr[bias], prev.dot(curr[weight]) + curr[bias]);
}
"""

IBP_CONSTRAINTFLOW = """
You are a formal methods expert working on neural network verification.
Your task is to generate the IBP transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

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
- func priority(Neuron n) = n[layer];

Don't add comments to DSL.
"""


prmpt_relu_ibp = """
def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

transformer ibp{
    Relu -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u])) : (((prev[u]) <= 0) ? (0, 0) : (0, (prev[u])));
}
"""

prmpt_abs_ibp = """
def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

transformer ibp{
    Abs -> (((prev[l]) >= 0) ? ((prev[l]), (prev[u])) : (((prev[u]) <= 0) ? (-prev[u], -prev[l]) : (0, max(-prev[l], prev[u]))));
}
"""

prmpt_affine_ibp = """
def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

transformer ibp{
    Affine -> ((prev.dot(curr[weight]) + curr[bias]).map(simplify_lower), (prev.dot(curr[weight]) + curr[bias]).map(simplify_upper));
}
"""


DEEPZ_CONSTRAINTFLOW = """
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepZ transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

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

Don't add comments to DSL.
"""


prmpt_relu_deepz = """
def Shape as (Float l, Float u, SymExp z){[(curr[l]<=curr),(curr[u]>=curr),(curr In curr[z])]};

transformer deepz{
    Relu -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev[z])) : (((prev[u]) <= 0) ? (0, 0, 0) : (0, (prev[u]), ((prev[u]) / 2) + (((prev[u]) / 2) * eps)));
}
"""

prmpt_abs_deepz = """
def Shape as (Float l, Float u, SymExp z){[(curr[l]<=curr),(curr[u]>=curr),(curr In curr[z])]};

transformer deepz{
    Abs -> ((prev[l]) >= 0) ?
                ((prev[l]), (prev[u]), (prev[z])) :
                (((prev[u]) <= 0) ?
                    (-(prev[u]), -(prev[l]), -(prev[z])) :
                    (0, max(-prev[l], prev[u]), ((max(-prev[l], prev[u])) / 2) + (((max(-prev[l], prev[u])) / 2) * eps)));
}
"""

prmpt_affine_deepz = """
def Shape as (Float l, Float u, SymExp z){[(curr[u]>=curr),(curr In curr[z]),(curr[l]<=curr)]};

transformer deepz{
    Affine -> ((prev.dot(curr[weight]) + curr[bias]).map(simplify_lower), (prev.dot(curr[weight]) + curr[bias]).map(simplify_upper), prev[z].dot(curr[weight]) + (curr[bias]));
}
"""

PRMPT_RELU_REASONING = """
ReLU(x) = max(0, x), which is piecewise linear:
- For x ≤ 0, ReLU(x) = 0
- For x ≥ 0, ReLU(x) = x

To construct a sound abstract transformer over input interval [l, u]:

1. Concrete Bounds:
   - If u ≤ 0 → ReLU(x) = 0 for all x ⇒ output interval = [0, 0]
   - If l ≥ 0 → ReLU(x) = x ⇒ output interval = [l, u]
   - If l < 0 < u:
     - Lower bound = 0
     - Upper bound = line connecting (l, 0) and (u, u)

2. Symbolic Bounds:
   - If input lower bound ≥ 0 ⇒ propagate symbolic bounds directly
   - If input upper bound ≤ 0 ⇒ output is zero ⇒ symbolic bound = constant 0
   - If input spans across 0:
     - Lower bound remains 0, since ReLU(x) is always ≥ 0
     - Upper bound is constructed using a linear interpolation:
       - Slope = u / (u - l)
       - Interpolated upper = slope * input
"""

PRMPT_ABS_REASONING = """
Abs(x) = |x|, which is piecewise linear:
- For x ≥ 0, Abs(x) = x
- For x ≤ 0, Abs(x) = -x

To construct a sound abstract transformer over input interval [l, u]:

1. Concrete Bounds:
   - If l ≥ 0 → input always ≥ 0 ⇒ Abs(x) = x ⇒ output interval = [l, u]
   - If u ≤ 0 → input always ≤ 0 ⇒ Abs(x) = -x ⇒ output interval = [-u, -l]
   - If l < 0 < u → input may cross 0:
     - Lower bound = 0 (since |x| ≥ 0)
     - Upper bound = max(|l|, |u|)

2. Symbolic Bounds:
   - Case 1: input ≥ 0
     - Symbolic lower = input lower bound
     - Symbolic upper = input upper bound
   - Case 2: input ≤ 0
     - Symbolic lower = negation of input upper bound
     - Symbolic upper = negation of input lower bound
   - Case 3: input spans zero (l < 0 < u)
     - Lower bound = 0
     - Upper bound:
        - The abs function is V-shaped. The upper bound should overapproximate this V-shape with a single straight line.
        - This line connects the points (l, -l) and (u, u), forming the tightest linear overapproximation.
        - Slope = (u + l) / (u - l)
"""

op_list = [
    "Abs",
    "Neuron_add",
    "Affine",
    "Avgpool",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "Neuron_max",
    "Maxpool",
    "Neuron_min",
    "Minpool",
    "Neuron_mult",
    "Relu",
    "Relu6",
]


DSL1_IBP = """
def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);

func abs(Float x) = x > 0 ? x : -x;

func max_lower(Neuron n1, Neuron n2) = n1[l]>=n2[l] ? n1[l] : n2[l];
func max_upper(Neuron n1, Neuron n2) = n1[u]>=n2[u] ? n1[u] : n2[u];

func min_lower(Neuron n1, Neuron n2) = n1[l]<=n2[l] ? n1[l] : n2[l];
func min_upper(Neuron n1, Neuron n2) = n1[u]<=n2[u] ? n1[u] : n2[u];

func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);

func priority(Neuron n) = n[layer];
"""

DSL2_IBP = """
flow(forward, priority, true, ibp);
"""

DSL1_DEEPZ = """
def Shape as (Float l, Float u, SymExp z){[(curr[l]<=curr),(curr[u]>=curr),(curr In curr[z])]};

func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);

func priority(Neuron n) = n[layer];

func abs(Float x) = x > 0 ? x : -x;

func s1(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
func i1(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (s1(x1, x2) * x1);
func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
func f2(Float x) = x * ((x + 3) / 6);

func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);

"""

DSL2_DEEPZ = """
flow(forward, priority, true, deepz);
"""

DSL1_DEEPPOLY = """
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
"""

DSL2_DEEPPOLY = """
flow(forward, priority, true, deeppoly);
"""


op_list_test = [
    "Abs",
]


class GlobalState:
    gen_rounds_now = 0
    repair_rounds_now = 0
    ce_number_now = 0
