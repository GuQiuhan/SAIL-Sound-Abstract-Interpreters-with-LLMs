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


def model_repair(client: Client, is_chat: bool, dsl: str, err: str) -> str:
    logging.info(f"\n💡 [Model Repair] Triggered model repair due to error:\n {err}")
    print("\n💡 [Model Repair] Triggered model repair due to error:\n%s", err)
    prompt = f"""You are a DSL repair assistant. Fix the following DSL code based on the error.
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
        fixed_code = model_repair(client, is_chat, fixed_code, syn_err)
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
        fixed_code = model_repair(client, is_chat, fixed_code, sem_err)
        GlobalState.repair_rounds_now += 1
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
transformer deeppoly{
    HardSigmoid ->
        ((prev[u]) <= -3) ? (0, 0, 0, 0) :
        (((prev[l]) >= 3) ? (1, 1, 1, 1) :
        ( ((prev[l]) >= -3) & ((prev[u]) <= 3) ?
            (0.1666666667 * prev[l] + 0.5, 0.1666666667 * prev[u] + 0.5, 0.1666666667 * prev + 0.5, 0.1666666667 * prev + 0.5 ) :
            ( (prev[l] < -3) & (prev[u] > 3) ?
                (0, 1, (prev - prev[l]) * (1 - 0) / (prev[u] - prev[l]) + 0, (prev - prev[l]) * (1 - 0) / (prev[u] - prev[l]) + 0)
                :
                (max(0, 0.1666666667 * prev[l] + 0.5), min(1, 0.1666666667 * prev[u] + 0.5), max(0, 0.1666666667 * prev + 0.5), min(1, 0.1666666667 * prev + 0.5))
            )
        ));
}
    """

    success, fixed_code = check("deeppoly", client, True, dsl)

    if success:
        print("✅ DSL is valid.\n", fixed_code)
    else:
        print("❌ Invalid DSL even after fix:\n", fixed_code)
