import os
import subprocess
import sys
import tempfile

from tabulate import tabulate

from constraintflow.core.verifier.provesound import run_verifier_from_str
from generation.request import Client, TGIClient
from generation.utils import *
from generation.validator.repair import check


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

    """

    validator = make_constraintflow_validator("deeppoly", client, True)

    result, ce, repaired_dsl = validator(dsl)
    print(result)
    print(ce)
    print(repaired_dsl)
