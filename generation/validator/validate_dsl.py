import subprocess
import tempfile
import os
import sys
from tabulate import tabulate


from constraintflow.experiments.experiments_correct import run_verifier_from_str

DSL1_IBP='''
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
'''

DSL2_IBP='''
flow(forward, priority, true, ibp);
'''

DSL1_DEEPZ='''
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

'''

DSL2_DEEPZ='''
flow(forward, priority, true, deepz);
'''

DSL1_DEEPPOLY='''
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
'''

DSL2_DEEPPOLY='''
flow(forward, priority, true, deeppoly);
'''

def make_constraintflow_validator(certifier: str):
    DSL1 = {
        "ibp": DSL1_IBP,
        "deepz": DSL1_DEEPZ,
        "deeppoly": DSL1_DEEPPOLY, 
    }

    DSL2 = {
        "ibp": DSL2_IBP,
        "deepz": DSL2_DEEPZ,
        "deeppoly": DSL2_DEEPPOLY,
    }

    if certifier not in DSL1 or certifier not in DSL2:
        raise ValueError(f"Unknown certifier: {certifier}")

    def validator(dsl: str):
        full_dsl = DSL1[certifier] + dsl + DSL2[certifier]
        return run_verifier_from_str(full_dsl)

    return validator


if __name__ == "__main__":

    dsl="""
transformer deeppoly{
    Relu6 -> 
        ((prev[l]) >= 6) ? 
            (6, 6, 6, 6) 
        : 
            (((prev[u]) <= 0) ? 
                (0, 0, 0, 0)
            : 
                ((prev[l] >= 0) ? 
                    (prev[l], min(prev[u], 6), prev, prev)
                : 
                    ((prev[u] <= 6) ? 
                        (0, prev[u], 0, ((prev[u] / (prev[u] - prev[l])) * prev) - ((prev[u] * prev[l]) / (prev[u] - prev[l])))
                    : 
                        (0, 6, 0, (((6 - prev[l]) / (prev[u] - prev[l])) * prev) - ((prev[l] * 6) / (prev[u] - prev[l])))
                    )
                )
            )
        ;
}
    """


    validator = make_constraintflow_validator("deeppoly")

    result, ce = validator(dsl)
    print(result)
    print(ce)


'''

transformer deeppoly{
    HardSigmoid -> 
        ((prev[u]) <= (0- 3.0)) ? (0, 0, 0, 0) : 
        ((prev[l]) >= 3) ? (1, 1, 1, 1) :
        (0,1,0,1);
}

The current DSL does not support negative floating-point constants directly.
To express a negative float like -3.0, it must be rewritten as an arithmetic expression, such as 0 - 3.0.
Otherwise, the parser will throw an error like no viable alternative at input, because -3.0 is not recognized as a valid token.
'''