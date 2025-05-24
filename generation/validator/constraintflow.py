import subprocess
import tempfile
import os
import sys


EXPERIMENTS_SCRIPT = "../../experiments/experiments_correct.py"

def constraintflow_validator(code: str, nprev: int = 4, nsymb: int = 4) -> bool:
    """
    Runs the ConstraintFlow verifier on the given DSL code.

    Args:
        code (str): The DSL transformer block to validate.
        nprev (int): Number of prev neurons.
        nsymb (int): Number of symbolic dimensions.

    Returns:
        bool: True if the DSL passes soundness and shape checks, False otherwise.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name
        print(tmp_path)

    print("here")


'''
    try:
        result = subprocess.run(
            ["python", EXPERIMENTS_SCRIPT, tmp_path, str(nprev), str(nsymb)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        print("--- ConstraintFlow Verifier Output ---")
        print(result.stdout)
        print(result.stderr)

        return result.returncode == 0 and "Certifier" in result.stdout

    except Exception as e:
        print(f"[Validator Error] {e}")
        return False

    #finally:
        #os.remove(tmp_path)
'''

if __name__ == "__main__":

    RELU6_DSL = """
Transformer DeepPoly(curr, prev){
ReLU6 -> prev[u] <= 0 ? (0, 0, 0, 0) : (
    prev[l] >= 6 ? (6, 6, 6, 6) : (
    prev[l] >= 0 && prev[u] <= 6 ? (prev[l], prev[u], prev, prev) : (
    prev[l] < 0 && prev[u] <= 6 ? (
        0,
        prev[u],
        0,
        ((prev[u] / (prev[u] - prev[l])) * (prev - prev[l]))
    ) : (
    prev[l] >= 0 && prev[u] > 6 ? (
        prev[l],
        6,
        prev,
        ((6 - prev[l]) / (prev[u] - prev[l])) * (prev - prev[u]) + 6
    ) : (
        0,
        6,
        0,
        ((6 / (prev[u] - prev[l])) * (prev - prev[l]))
    ))));
}
"""
    valid = constraintflow_validator(RELU6_DSL, nprev=4, nsymb=4)
    print(valid)