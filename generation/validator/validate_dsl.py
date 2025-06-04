import subprocess
import tempfile
import os
import sys
from tabulate import tabulate


from constraintflow.experiments.experiments_correct import run_verifier_from_str

def constraintflow_validator(dsl:str):
    return run_verifier_from_str(dsl)

if __name__ == "__main__":

    dsl1="""
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
    Relu6 -> ((prev[l]) >= 0)? ((prev[l]), (prev[u]), (prev), (prev), (prev), (prev)) : (((prev[u]) <= 0)? (0-(prev[u]), 0-(prev[l]), 0-(prev), 0-(prev), 0-(prev), 0-(prev)) : (0, max(prev[u], 0-prev[l]), prev, prev*(prev[u]+prev[l])/((prev[u]-prev[l]) + 1), prev*(prev[u]+prev[l])/((prev[u]-prev[l]) + 1), prev*(prev[u]+prev[l])/((prev[u]-prev[l]) + 1), prev*(prev[u]+prev[l])/((prev[u]-prev[l]) + 1), prev*(prev[u]+prev[l])/((prev[u]-prev[l]) + 1)));
}

flow(forward, priority, true, deeppoly);    
    """


    result, ce = run_verifier_from_str(dsl1)
    print(result)
    print(ce)