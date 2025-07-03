import re
from typing import Optional, Tuple

from semantics_check import check_semantic
from syntax_check import SyntaxChecker

from generation.request import Client, TGIClient

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
            return code
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
    while syntax_attempt < MAX_RETRIES:
        syn_result, fixed_code, syn_err = syntax_checker.check(fixed_code)
        if syn_result:
            break
        fixed_code = model_repair(client, is_chat, fixed_code, syn_err)
        fixed_code = make_block_extractor(certifier, fixed_code)
        syntax_attempt += 1

    if not syn_result:
        return False, ""

    # ---- Semantic Repair Phase ----
    semantic_attempt = 0
    while semantic_attempt < MAX_RETRIES:
        sem_result, _, sem_errs = check_semantic(fixed_code)
        if sem_result:
            return True, fixed_code
        sem_err = "\n".join(sem_errs)
        fixed_code = model_repair(client, is_chat, fixed_code, sem_err)
        fixed_code = make_block_extractor(certifier, fixed_code)
        semantic_attempt += 1

    return False, ""


if __name__ == "__main__":
    dsl = """
    transformer deeppoly{
        HardSwish ->
        ((prev[l] >= 3)
            ? (prev[l], prev[u], prev, prev)
            : ((prev[u] <= -3)
                ? (0, 0, 0, 0)
                : ( (0,
                      max( max(prev[u] * ((prev[u] + 3) / 6), prev[l] * ((prev[l] + 3) / 6)), prev[l] * ((prev[u] + 3) / 6) ),
                      prev * ((prev + 3) / 6),
                      prev * ((prev + 3) / 6)
                  )
              )
          )
    );}
    """

    success, fixed_code, error = check(dsl)
    if success:
        print("✅ DSL is valid.")
    else:
        print("❌ DSL has errors:\n", error)
        print("\n🔧 Fixed DSL:\n", fixed_code)
