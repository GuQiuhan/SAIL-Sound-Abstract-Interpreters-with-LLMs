import os
import subprocess
import sys
import tempfile

from tabulate import tabulate
from validator.repair import check

from constraintflow.experiments.experiments_correct import run_verifier_from_str
from generation.request import Client, TGIClient

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


def make_constraintflow_validator(certifier: str, client: Client, is_chat: bool):
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
        """
        Returns:
         -(True, ""): succeed
         -(False, str): fail, counterexample
         -(False, ""): fail, invalid, no counterexample
        """
        success, repaired_dsl = check(certifier, client, is_chat, dsl)
        if not success:
            return False, "", repaired_dsl  # invalid, no counterexample

        full_dsl = DSL1[certifier] + repaired_dsl + DSL2[certifier]
        result, ce = run_verifier_from_str(full_dsl)  # return (T/F, ce: str/"")
        return result, ce, repaired_dsl

    return validator


if __name__ == "__main__":
    client = TGIClient(model="http://ggnds-serv-01.cs.illinois.edu:8084")

    dsl = """
transformer deeppoly{
    HardTanh ->
      ((prev[u]) <= -1) ? (-1, -1, -1, -1) :
      ((prev[l]) >= 1) ? (1, 1, 1, 1) :
      (((prev[l]) >= -1) & ((prev[u]) <= 1)) ? (prev[l], prev[u], prev[L], prev[U]) :
      (((prev[l]) < -1) & ((prev[u]) <= 1)) ? (-1, prev[u], -1, prev[U]) :
      (((prev[l]) >= -1) & ((prev[u]) > 1)) ? (prev[l], 1, prev[L], 1) :
      ((prev[l]) < -1 & (prev[u]) > 1) ? (-1, 1, -1, 1) :
      (prev[l], prev[u], prev[L], prev[U]);
}
    """

    validator = make_constraintflow_validator("deeppoly", client, True)

    result, ce = validator(dsl)
    print(result)
    print(ce)
