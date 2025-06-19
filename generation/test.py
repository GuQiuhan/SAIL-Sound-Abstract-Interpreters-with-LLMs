# used to test reasoning capability


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



from time import time
import logging
import re
import shutil
from datetime import datetime
from typing import Callable, List, Optional
import traceback


from request import Client
from abc import ABC, abstractmethod

from validator.validate_dsl import *

from utils import *


class Step:
    def __init__(self, prompter, composer=None, eos=None, validator=None):
        self.prompter: Callable[[Optional[str]], Union[str, List[Dict[str, str]]]] = prompter  # unified for both prompt and chat models
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[Union[str, List[Dict[str, str]]], str, str], Union[str, bool]] = composer
        self.validator: Callable[[str], None] = validator  # validate the code
        # add augmentation prompting
        self.aug_prompt = ""
        self.error_generation = ""  # List to store error generation
        self.counter_example = ""  # List to store counter examples
        self.given_code = ""  # provided code

    def set_augmentation_prompt(self, aug_prompt: str, error_generation: str, counter_example: str):
        self.aug_prompt = aug_prompt
        self.error_generation = error_generation
        self.counter_example = counter_example

    def prompter_with_augmentation(self, old_prmpt: str) -> str:
        """Generates the prompt with augmentation, error generation and counter examples."""
        augmented_prompt = old_prmpt
        if self.aug_prompt:
            last_api_index = augmented_prompt.rfind("API: ")
            error_index = augmented_prompt.find("Error Generation:", last_api_index)
            if error_index != -1:
                augmented_prompt = (
                    augmented_prompt[:error_index]
                    + "- "
                    + self.aug_prompt
                    + "\n"
                    + augmented_prompt[error_index:]
                )
            
            last_api_index = augmented_prompt.rfind("API: ")
            generation_index = augmented_prompt.find("Counter Example:", last_api_index)
            if generation_index != -1:
                augmented_prompt = (
                    augmented_prompt[:generation_index]
                    + self.error_generation
                    + "\n"
                    + augmented_prompt[generation_index:]
                )

            last_api_index = augmented_prompt.rfind("API: ")
            ce_index = augmented_prompt.rfind("Generation:", last_api_index)
            if ce_index != -1:
                augmented_prompt = (
                    augmented_prompt[:ce_index]
                    + self.counter_example
                    + "\n"
                    + augmented_prompt[ce_index:]
                )
            
        return augmented_prompt


def step_by_step_gen(client: Client, steps: List[Step], is_chat: bool):
    """
    Executes a sequence of steps for DSL generation.
    Step 1: DSL transformer generation (returns DSL code)

    Returns:
        (success: bool, result: str, error_message: str)
    """
    
    code = "" # final code after all steps

    for index, step in enumerate(steps, start=1):
        logging.info(f"[STEP {index}] Starting step {index}/{len(steps)}")
        retry_count = 0
        success = False

        new_code = "" # final code after this step

        while retry_count < 2 and not success:
            messages_or_prompt = step.prompter(code)
            #prompt = step.prompter_with_augmentation(prompt) # need to unify them

            print(f"[STEP {index}] Prompting: :\n{messages_or_prompt}")


            completions = [
                client.chat(messages=messages_or_prompt) if is_chat
                else client.textgen(prompt=messages_or_prompt)
                for _ in range(3)
            ] # multiple(3) samples

            for sample_id, completion in enumerate(completions, start=1):
                if "Model Generation Error" in completion:
                    logging.warning(f"[STEP {index}] Sample {sample_id}: Model Generation Error")
                    continue
                
                new_code = step.composer(messages_or_prompt, completion, code) # update code here
                #print(f"[STEP {index}] Sample {sample_id}: Completion:\n{completion}")
                #print(f"[STEP {index}] Sample {sample_id}: Parsed DSL:\n{code}")

                if step.validator:
                    try:
                        result, ce = step.validator(code)
                    except Exception as e:
                        #logging.warning(f"[STEP {index}] Sample {sample_id}: Validator exception: {e}. Code: \n {code}\n")
                        logging.warning(f"[STEP {index}] Sample {sample_id}: Validator exception. Full traceback:\n{traceback.format_exc()}\nCode:\n{code}\n")

                        result, ce = False, None

                    if result:
                        success = True
                        code = new_code
                        logging.info(f"[STEP {index}] Sample {sample_id}: Validation passed.")
                        break
                    else:
                        logging.info(f"[STEP {index}] Sample {sample_id}: Validation failed.")
                else:
                    success = True
                    code = new_code
                    break

            if not success:
                retry_count += 1
                logging.info(
                    f"[STEP {index}] All {len(completions)} samples failed validation. Retrying {retry_count}/2..."
                )


        if not success:
            return False, new_code, f"[STEP {index}] Failed after 2 retries." # return the failed code

    return True, code, ""



        
if __name__ == "__main__":
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
        "--log-dir", help="Path to the log directory", default="reasoninglogs/", required=False
    )

    parser.add_argument(
        "--model","-m",
        type=str,
        required=False,
        default="deepseek",
        help="Model keyword to select from model-port map. E.g., deepseek, llama-4, gpt-4.1"
    )

    parser.add_argument(
        "--certifier","-c",
        type=str,
        required=False,
        choices=["deeppoly", "ibp", "deepz"],
        default="deeppoly",
        help="Certifier type: deeppoly, ibp, deepz"
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("tmplogs", run_timestamp)
    result_dir = os.path.join(run_dir, "results")
    log_path = os.path.join(run_dir, "generation.log")


    model_keyword = args.model.lower()
    certifier = args.certifier

    if model_keyword not in PORT_MAP:
        raise ValueError(f"Model '{args.model}' not found in PORT_MAP.")

    MODEL_ENDPOINTS = {
        model_keyword: PORT_MAP[model_keyword]
    }

    # @qiuhan: TODO: allow multiple models

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

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

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
        overall_start_time = time()

        for op_name in p.track(sorted(["join", "meet"])):
            op_start_time = time()
            doc = {"api": op_name}

            logging.info(f"{datetime.now()} - Extracting {doc['api']}")

            for model_name, url in MODEL_ENDPOINTS.items():
                model_out_dir = os.path.join(result_dir, model_name)
                success_dir = os.path.join(model_out_dir, "success")
                failure_dir = os.path.join(model_out_dir, "failure")
                api_name = doc["api"]

                logging.info(f"\nAPI: {api_name} -> Model: {model_name} @ {url}")
                client = TGIClient(model=url, max_new_tokens=2048)


                def generate_dsl(api, dsl=None, debug=False) -> str:
                    steps = []
                    model_type = "prompt" if "deepseek" in args.model.lower() else "chat"
                    is_chat = model_type == "chat"

                    def make_block_extractor(certifier: str):
                        keyword = certifier.lower()  # "deeppoly", "ibp", "deepz"

                        def extract_constraintflow_block(prmpt, cmpl, code) -> str:
                            """
                            Extract everything starting from the correct transformer keyword (deeppoly, ibp, deepz)
                            until the closing brace '}' that balances the opening one.
                            """
                            match = re.search(rf'({re.escape(keyword)}\s*\{{)', cmpl)
                            if not match:
                                return ""

                            start_idx = match.start()
                            brace_count = 0
                            for i in range(start_idx, len(cmpl)):
                                if cmpl[i] == '{':
                                    brace_count += 1
                                elif cmpl[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        return "transformer "+ cmpl[start_idx:i+1].strip()
                            
                            return "transformer "+cmpl[start_idx:].strip()

                        return extract_constraintflow_block


                    extractor = make_block_extractor(certifier)
                    validation = make_constraintflow_validator(certifier)

                    def c (a,b,c): return b


                    if is_chat:
                        def chat_prompter(code: Optional[str]) -> List[dict]: # `code` here means the code that is generated last time
                            return [
                                {"role": "system", "content": f"{CONSTRAINTFLOW_SYSTEM_PROMPT}"},
                                {"role": "user", "content": "Reason the transformer for `relu` operator "},
                                {"role": "assistant", "content": f"{PRMPT_RELU_REASONING}"},
                                {"role": "user", "content": "Reason the transformer for `abs` operator "},
                                {"role": "assistant", "content": f"{PRMPT_ABS_REASONING}"},
                                {"role": "user", "content": f"Reason the transformer for `{api}` operator "},
                            ]
                        prompter1 = chat_prompter
                    else:
                        def prompt_prompter(code: Optional[str]) -> str:
                            return f"""
{CONSTRAINTFLOW_SYSTEM_PROMPT}

### Example: ReLU operator
Input: Reason the transformer for `relu` operator
Reasoning: {PRMPT_RELU_REASONING}
Output:
{prmpt_relu}

### Example: Abs operator
Input: Generate the transformer for `abs` operator
Reasoning: {PRMPT_ABS_REASONING}
Output:
{prmpt_abs}


### Now generate the transformer for `{api}` operator
Input: Generate the transformer for `{api}` operator
Reasoning:
"""
                        prompter1 = prompt_prompter


                    steps.append(
                        Step(
                            prompter=prompter1,
                            composer=c,
                            eos=["\n# END"],
                            validator= None,  
                        )
                    )

                    if is_chat:
                        def chat_prompter(code: Optional[str]) -> List[dict]:
                            return [
                                {"role": "system", "content": f"{CONSTRAINTFLOW_SYSTEM_PROMPT}"},
                                {"role": "user", "content": f"Generate the transformer for `relu` operator based on the reasoning: {PRMPT_RELU_REASONING}"},
                                {"role": "assistant", "content": prmpt_relu},
                                {"role": "user", "content": f"Generate the transformer for `abs` operator based on the reasoning: {PRMPT_ABS_REASONING}"},
                                {"role": "assistant", "content": prmpt_abs},
                                {"role": "user", "content": f"Generate the transformer for `{api}` operator based on the reasoning: {code}"},
                            ]
                        prompter2 = chat_prompter
                    else:
                        def prompt_prompter(code: Optional[str]) -> str:
                            return f"""
{CONSTRAINTFLOW_SYSTEM_PROMPT}

### Example: ReLU operator
Input: Reason the transformer for `relu` operator
Reasoning: {PRMPT_RELU_REASONING}
Output:
{prmpt_relu}

### Example: Abs operator
Input: Generate the transformer for `abs` operator
Reasoning: {PRMPT_ABS_REASONING}
Output:
{prmpt_abs}


### Now generate the transformer for `{api}` operator
Input: Generate the transformer for `{api}` operator
Reasoning:
"""
                        prompter2 = prompt_prompter
                    
                    steps.append(
                        Step(
                            prompter=prompter2,
                            composer=c,
                            eos=["\n# END"],
                            validator= None,  
                        )
                    )

                    return step_by_step_gen(client, steps, is_chat)

                result, code, error = generate_dsl(doc["api"])  
                op_end_time = time()  
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

        overall_end_time = time()
        total_runtime = overall_end_time - overall_start_time
        logging.info(f"✅ Total runtime for all operators: {total_runtime:.2f} seconds")