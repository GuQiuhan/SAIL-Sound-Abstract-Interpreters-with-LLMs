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




def constraintflow_validator(dsl:str):
    dsl= DSL1_DEEPZ+dsl+DSL2_DEEPZ
    return run_verifier_from_str(dsl)

if __name__ == "__main__":

    dsl1="""
transformer deepz{
    Abs -> ((prev[l]) >= 0) ? 
                ((prev[l]), (prev[u]), (prev[z])) : 
                (((prev[u]) <= 0) ? 
                    (-(prev[u]), -(prev[l]), -(prev[z])) : 
                    (0, max(-prev[l], prev[u]), ((max(-prev[l], prev[u])) / 2) + (((max(-prev[l], prev[u])) / 2) * eps)));
}
    """


    result, ce = constraintflow_validator(dsl1)
    print(result)
    print(ce)