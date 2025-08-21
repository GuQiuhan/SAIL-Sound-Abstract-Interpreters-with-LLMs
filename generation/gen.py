"""
DSL Transformer Generation Script
-----------------------------------------------------------

This script automatically generates ConstraintFlow-style DSL transformers
for deep learning operators.

1. Supports multiple abstract domains (certifiers: DeepPoly, IBP, DeepZ)
and multiple models (e.g., DeepSeek, GPT-4, LLaMA).
2. Automatically detects whether the model is chat-based (e.g., DeepSeek) or prompt-based.
3. Extracts the DSL transformer block from generated output based on the certifier type.
4. Validates generated code using formal verification tools.
5. Logs successful and failed generations; saves results with timestamps.

Usage:
    python gen_dsl_transformer.py --model deepseek --certifier deeppoly
    (Optional: --log-dir <log_folder> --output-dir <output_folder>)

Output:
    Results are saved under logs/<timestamp>/results/ by default.
"""


import json
import logging
import re
import shutil
import traceback
import typing
from abc import ABC, abstractmethod
from datetime import datetime
from statistics.rounds import *
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from evaluator.eval import *
from request import Client
from utils import *
from validator.soundness_check import *


class Step:
    def __init__(
        self, prompter, composer=None, eos=None, validator=None, evaluator=None
    ):
        self.prompter: Callable[
            [Optional[str]], Union[str, List[Dict[str, str]]]
        ] = prompter  # unified for both prompt and chat models

        self.eos: List[str] = eos or []

        # (prompt, completion, old code) = composer => new code
        self.composer: Callable[[str, str, str], Union[str, bool]] = composer
        self.validator: Callable[
            [str], Tuple[bool, Optional[str], Optional[str]]
        ] = validator  # validate the code
        self.evaluator: Callable[[str], float] = evaluator  # New: code -> score

        # add augmentation prompting
        self.aug_prompt = ""
        self.error_generation = ""  # List to store error generation
        self.counter_example = ""  # List to store counter examples

    def save_failure(self, error_generation: str, ce: str):
        # just save one each time
        self.error_generation = ""
        self.counter_example = ""
        self.error_generation = self.error_generation + "\n" + error_generation + "\n"
        self.counter_example = self.counter_example + "\n" + ce + "\n"

    def prompter_with_augmentation(
        self, old_prmpt: typing.Union[str, typing.List[typing.Dict[str, str]]]
    ) -> typing.Union[str, typing.List[typing.Dict[str, str]]]:
        """
        Augments the original prompt with:
        - The previously generated (invalid) code
        - The counterexample from the validator
        Supports both chat-format and plain-text prompts.
        """
        if not self.counter_example and not self.error_generation:
            return old_prmpt  # Nothing to add

        ce_note = (
            f"\n\n# Previously generated (invalid) code:\n{self.error_generation}\n\n"
            f"# Counter Example respectively:\n{self.counter_example}\n"
            f"# Learn from the failed generation above and revise your output accordingly. Output the DSL only."
        )

        self.counter_example = ""  # clear after augmentation
        self.error_generation = ""

        if isinstance(old_prmpt, list):  # Chat format
            for msg in reversed(old_prmpt):
                if msg.get("role") == "user":
                    msg["content"] += ce_note
                    break
            return old_prmpt

        elif isinstance(old_prmpt, str):  # Text prompt format
            return old_prmpt + ce_note

        return old_prmpt


def step_by_step_gen(client: Client, steps: List[Step], is_chat: bool):
    """
    Executes a sequence of steps for DSL generation.
    Step 1: DSL transformer generation (returns DSL code)

    Returns:
        (success: bool, result: str, error_message: str)
    """

    for index, step in enumerate(steps, start=1):
        logging.info(f"[STEP {index}] Starting step {index}/{len(steps)}")
        retry_count = 0
        success = False

        code = ""
        best_score = float("inf")  # start to search
        best_code = ""  # sound or unsound

        while retry_count < MAX_RETRIES and not success:
            GlobalState.gen_rounds_now += 1

            code = ""
            messages_or_prompt = step.prompter(code)
            messages_or_prompt = step.prompter_with_augmentation(messages_or_prompt)

            print(f"[STEP {index}] Messages_or_prompt: \n {messages_or_prompt}\n")

            completions = [
                client.chat(messages=messages_or_prompt)
                if is_chat
                else client.textgen(prompt=messages_or_prompt)
                for _ in range(3)
            ]  # multiple(3) samples

            # at most will save 1 wrong generations -> better than current best_code

            for sample_id, completion in enumerate(completions, start=1):
                if "Model Generation Error" in completion:
                    logging.warning(
                        f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Model Generation Error"
                    )
                    continue
                code = step.composer("", completion, code)

                print(
                    f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Completion:\n{completion}\n"
                )
                print(
                    f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Parsed DSL:\n{code}\n"
                )

                if code == "":
                    logging.warning(
                        f"[STEP {index}] Sample {sample_id}: No valid generation: \n{completion}\n"
                    )
                    continue

                if step.validator:
                    result = False
                    try:
                        result, ce, code = step.validator(code)
                    except Exception as e:
                        logging.warning(
                            f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Validator exception. Full traceback:\n{traceback.format_exc()}\nCode:\n{code}\n"
                        )

                        result, ce = False, ""

                    if result:
                        success = True
                        best_code = code
                        logging.info(
                            f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Validation passed for code: \n{best_code}."
                        )
                        return True, best_code, ""
                    else:  # TODO: augment the prompt with ce. Done.
                        if ce:
                            # if have a ce, get the score first
                            logging.info(
                                f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Validation failed. Get counter example: \n {ce}.\n Start to evaluate the deviation."
                            )

                            score = step.evaluator(code)
                            if score < best_score:
                                logging.info(
                                    f"best_score : score = {best_score} : {score}",
                                )

                                best_score = score  # update
                                best_code = code

                                step.save_failure(code, ce)
                                GlobalState.ce_number_now += 1

                                logging.info(
                                    f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Get a 'better' unsound abstract transformer: \n{code}\n with the score {score}. Use it to guide the regeneration."
                                )

                        else:
                            logging.info(
                                f"[RETRY {retry_count} STEP {index}] Sample {sample_id}: Validation failed. Get no counter example. Other errors(semantic/syntactic) exist."
                            )
                else:
                    success = True
                    best_code = code
                    break

            if not success:
                retry_count += 1
                logging.info(
                    f"[RETRY {retry_count} STEP {index}] All {len(completions)} samples failed validation. Retrying {retry_count}/{MAX_RETRIES}..."
                )

            else:
                break

        if not success:
            return (
                False,
                best_code,
                f"[STEP {index}] Failed after {MAX_RETRIES} retries.",
            )
        else:
            break

    return True, best_code, ""


if __name__ == "__main__":
    # global gen_rounds_now
    # global repair_rounds_now
    # global ce_number_now

    import argparse
    import os

    import yaml
    from request import TGIClient
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="Path to save generation",
        default="results/",
        required=False,
    )
    parser.add_argument(
        "--log-dir", help="Path to the log directory", default="logs/", required=False
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=False,
        nargs="+",  # multiple models
        choices=[
            "llama",
            "llama-3.3",
            "llama-4",
            "deepseek",
            "gpt-4o",
            "gpt-4.1",
            "o4-mini",
            "gpt-5",
            "gpt-oss",
            "jamba",
            "titan",
            "nova",
            "claude",
            "mistral",
        ],
        default=["gpt-4o"],
        help="Model keyword to select from model-port map. E.g., deepseek, llama-4, gpt-4.1, gpt-4o",
    )

    parser.add_argument(
        "--certifier",
        "-c",
        type=str,
        required=False,
        nargs="+",  # multiple certifiers
        choices=["deeppoly", "ibp", "deepz"],
        default=["deeppoly"],
        help="Certifier type: deeppoly, ibp, deepz",
    )
    args = parser.parse_args()

    statistic = {}

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", run_timestamp)
    results_dir = os.path.join(run_dir, "results")

    if isinstance(args.model, str):
        model_keywords = [args.model.lower()]
    else:
        model_keywords = [m.lower() for m in args.model]

    if isinstance(args.certifier, str):
        certifiers = [args.certifier.lower()]
    else:
        certifiers = [c.lower() for c in args.certifier]

    for certifier in certifiers:
        result_dir = os.path.join(results_dir, certifier)

        # @qiuhan: TODO: allow multiple models. Done.
        MODEL_ENDPOINTS = {}
        for model_keyword in model_keywords:
            matched = False
            for model_name, endpoint in PORT_MAP.items():
                if model_keyword in model_name.lower():
                    MODEL_ENDPOINTS[model_name] = endpoint
                    matched = True
            if not matched:
                raise ValueError(f"No models matched keywords: {model_keyword}. ")

        for model_name in MODEL_ENDPOINTS:
            model_out_dir = os.path.join(result_dir, model_name)
            if os.path.exists(model_out_dir):
                shutil.rmtree(model_out_dir)
            os.makedirs(os.path.join(model_out_dir, "success"), exist_ok=True)
            os.makedirs(os.path.join(model_out_dir, "failure"), exist_ok=True)

        if certifier == "deeppoly":
            CONSTRAINTFLOW_SYSTEM_PROMPT = DEEPPOLY_CONSTRAINTFLOW
            prmpt_relu = prmpt_relu_deeppoly
            prmpt_abs = prmpt_abs_deeppoly
            prmpt_affine = prmpt_affine_deeppoly
        elif certifier == "ibp":
            CONSTRAINTFLOW_SYSTEM_PROMPT = IBP_CONSTRAINTFLOW
            prmpt_relu = prmpt_relu_ibp
            prmpt_abs = prmpt_abs_ibp
            prmpt_affine = prmpt_affine_ibp
        elif certifier == "deepz":
            CONSTRAINTFLOW_SYSTEM_PROMPT = DEEPZ_CONSTRAINTFLOW
            prmpt_relu = prmpt_relu_deepz
            prmpt_abs = prmpt_abs_deepz
            prmpt_affine = prmpt_affine_deepz
        else:
            raise ValueError(f"Unknown certifier: {certifier}")

        progress_bar = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

        prefix = os.path.join(os.path.dirname(__file__), "prompt/prompts")

        with progress_bar as p:
            for model_name, endpoint_info in MODEL_ENDPOINTS.items():
                url = endpoint_info["url"]
                model_type = endpoint_info["mode"]

                statistic = {}
                overall_start_time = time.time()

                model_out_dir = os.path.join(result_dir, model_name)
                success_dir = os.path.join(model_out_dir, "success")
                failure_dir = os.path.join(model_out_dir, "failure")

                statistic_dir = os.path.join(model_out_dir, "statistics")
                os.makedirs(statistic_dir, exist_ok=True)
                statistic_path = os.path.join(statistic_dir, "statistic.json")

                log_path = os.path.join(model_out_dir, "generation.log")
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                logging.basicConfig(
                    filename=log_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode="a",
                )

                client = TGIClient(model=url, max_new_tokens=2048)

                if model_name not in statistic:
                    statistic[model_name] = []

                for op_name in p.track(sorted(op_list)):  # TODO: change the opt list
                    op_start_time = time.time()
                    doc = {"api": op_name}

                    logging.info(f"{datetime.now()} - Extracting {doc['api']}")
                    api_name = doc["api"]

                    GlobalState.gen_rounds_now = 0
                    GlobalState.repair_rounds_now = 0
                    GlobalState.ce_number_now = 0

                    logging.info(f"\nAPI: {api_name} -> Model: {model_name} @ {url}")

                    def generate_dsl(api, model, dsl=None, debug=False) -> str:
                        steps = []

                        is_chat = model_type == "chat"

                        def make_block_extractor(certifier: str, api: str):
                            """
                            Search for patterns of the form:
                                transformer <keyword> { ... <api> ... }
                            - <keyword> is provided by the certifier
                            - Whitespace/newlines may vary
                            - <api> must be matched as a whole word
                            - Curly braces {} must be balanced; otherwise, return an empty string
                            - If no block contains the <api>, return an empty string
                            - If multiple blocks match, return the LAST matching block
                            """
                            keyword = certifier.lower()  # "deeppoly", "ibp", "deepz"

                            start_re = re.compile(
                                rf"transformer\s+{re.escape(keyword)}\s*\{{",
                                re.IGNORECASE | re.DOTALL,
                            )

                            api_pat = re.compile(
                                rf"\b{re.escape(api)}\b", re.IGNORECASE
                            )

                            def extract_constraintflow_block(
                                prmpt, cmpl: str, code
                            ) -> str:
                                pos = 0
                                n = len(cmpl)
                                last_match_block = ""

                                while True:
                                    m = start_re.search(cmpl, pos)
                                    if not m:
                                        return last_match_block

                                    start_idx = m.start()  # 'transformer'
                                    brace_start = m.end() - 1  # '{'

                                    brace_count = 0
                                    i = brace_start
                                    block_end = None
                                    while i < n:
                                        ch = cmpl[i]
                                        if ch == "{":
                                            brace_count += 1
                                        elif ch == "}":
                                            brace_count -= 1
                                            if brace_count == 0:
                                                block_end = i + 1
                                                break
                                        i += 1

                                    if block_end is None:
                                        return ""

                                    block_text = cmpl[start_idx:block_end].strip()

                                    if api_pat.search(block_text):
                                        last_match_block = block_text

                                    pos = block_end

                            return extract_constraintflow_block

                        extractor = make_block_extractor(certifier, api)
                        validation = make_constraintflow_validator(
                            certifier, client, is_chat
                        )
                        evaluator = make_constraintflow_evaluator(certifier)

                        if is_chat:

                            def chat_prompter(code: Optional[str]) -> List[dict]:
                                return [
                                    {
                                        "role": "system",
                                        "content": f"{CONSTRAINTFLOW_SYSTEM_PROMPT}",
                                    },
                                    {
                                        "role": "user",
                                        "content": "Generate the transformer for `relu` operator ",
                                    },
                                    {"role": "assistant", "content": prmpt_relu},
                                    {
                                        "role": "user",
                                        "content": "Generate the transformer for `abs` operator ",
                                    },
                                    {"role": "assistant", "content": prmpt_abs},
                                    {
                                        "role": "user",
                                        "content": "Generate the transformer for `affine` operator ",
                                    },
                                    {"role": "assistant", "content": prmpt_affine},
                                    {
                                        "role": "user",
                                        "content": f"Generate the transformer for `{api}` operator ",
                                    },
                                ]

                            prompter = chat_prompter
                        else:

                            def prompt_prompter(code: Optional[str]) -> str:
                                return f"""
    {CONSTRAINTFLOW_SYSTEM_PROMPT}

    ### Example: ReLU operator
    Input: Generate the transformer for `relu` operator
    Reasoning: {PRMPT_RELU_REASONING}
    Output:
    {prmpt_relu}

    ### Example: Abs operator
    Input: Generate the transformer for `abs` operator
    Output:
    {prmpt_abs}

    ### Example: Affine operator
    Input: Generate the transformer for `affine` operator
    Output:
    {prmpt_affine}

    ### Now generate the transformer for `{api}` operator
    Input: Generate the transformer for `{api}` operator
    Output:
    """

                            prompter = prompt_prompter

                        steps.append(
                            Step(
                                prompter=prompter,
                                composer=extractor,
                                eos=["\n# END"],
                                validator=validation,
                                evaluator=evaluator,
                            )
                        )

                        return step_by_step_gen(client, steps, is_chat)

                    result, code, error = generate_dsl(doc["api"], model_name)
                    op_end_time = time.time()
                    op_time = op_end_time - op_start_time
                    logging.info(f"[{op_name}] Runtime: {op_time:.2f} seconds")

                    if not result:
                        target_path = os.path.join(failure_dir, f"{doc['api']}.txt")
                        with open(target_path, "w") as f:
                            f.write(code)
                        logging.error(
                            f"Failed with Error:{error}\n during generating code:\n{code}\n"
                        )
                    else:
                        target_path = os.path.join(success_dir, f"{doc['api']}.txt")
                        with open(target_path, "w") as f:
                            f.write(code)
                        logging.info(f"Succeed. Saved to {target_path}\n")

                    statistic[model_name].append(
                        [
                            op_name,
                            GlobalState.gen_rounds_now,
                            GlobalState.repair_rounds_now,
                            GlobalState.ce_number_now,
                            op_time,
                            bool(result),
                        ]
                    )
                overall_end_time = time.time()
                total_runtime = overall_end_time - overall_start_time
                statistic[model_name].append(
                    [total_runtime]
                )  # the last list just have one element
                logging.info(
                    f"✅ Total runtime for all operators with the model {model_name}: {total_runtime:.2f} seconds"
                )

                with open(statistic_path, "w") as f:
                    json.dump(statistic, f, indent=2)

                draw_all(statistic, statistic_dir)
