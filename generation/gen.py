"""
## Few-shot Prompting for Long DSL Generation
"""

import logging
import re
import shutil
from datetime import datetime
from typing import Callable, List, Optional


from request import Client
from abc import ABC, abstractmethod

from validator.validate_dsl import constraintflow_validator

# set max retry time every turn
MAX_RETRIES = 10

MODEL_ENDPOINTS = {
    #"gemma-7b": "http://10.192.122.120:8082",
    #"deepseek-6.7b": "http://10.192.122.120:8083",
    "llama3-1B": "http://ggnds-serv-01.cs.illinois.edu:8080",
    #"llama3-70B": "http://10.192.122.120:8086",
}

CONSTRAINTFLOW = """
DeepPoly certifier uses four kinds of bounds to approximate the operator: (Float l, Float u, PolyExp L, PolyExp U).
They must follow the constraints that: curr[l] <= curr <= curr[u] & curr[L] <= curr <= curr[U]. `curr` here means the current neuron, `prev` means the inputs to the operator.
So every transformer in each case of the case analysis must return four values.
"""

prmpt_relu= """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Relu -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev), (prev)) : (((prev[u]) <= 0) ? (0, 0, 0, 0) : (0, (prev[u]), 0, (((prev[u]) / ((prev[u]) - (prev[l]))) * (prev)) - (((prev[u]) * (prev[l])) / ((prev[u]) - (prev[l]))) ));
} 
"""

prmpt_abs = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Abs -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev), (prev)) : (((prev[u]) <= 0) ? (0-(prev[u]), 0-(prev[l]), 0-(prev), 0-(prev)) : (0, max(prev[u], 0-prev[l]), prev, prev*(prev[u]+prev[l])/(prev[u]-prev[l]) - (((2*prev[u])*prev[l])/(prev[u]-prev[l]))) );
}
"""



def gen(client: Client):
    """
    DSL transformer generation

    Returns:
        (success: bool, result: str, error_message: str)
    """
    
    for index, step in enumerate(steps, start=1):
        logging.info(f"[STEP {index}] Starting step {index}/{len(steps)}")
        retry_count = 0
        success = False

        while retry_count < MAX_RETRIES and not success:
            try:
                code = ""
                prompt = step.prompter(code)
                prompt = step.prompter_with_augmentation(prompt)

                

                completion = client.textgen(prompt=prompt)


                print("here")
                print(completion)

                if "Model Generation Error" in completion:
                    return False, "", f"[STEP {index}] Model Generation Error during completion."

                code = step.composer(prompt, completion, code)
                print(f"--- STEP {index} COMPLETION ---\n{completion}\n")
                print(f"--- STEP {index} PARSED RESULT ---\n{code}\n")

                # Step 1 returns bool
                if index == 1 and isinstance(result, bool):
                    if result:
                        logging.info("[STEP 1] Operator is supported by DeepPoly.")
                        success = True
                        code = ""  # reset for step 2
                    else:
                        return False, "", "[STEP 1] Operator is not supported by DeepPoly. Skipping transformer generation."
                # Step 2
                elif index==2:
                    if step.validator:
                        result, ce = step.validator(code)
                        if result:
                            success = True
                            logging.info(f"Validation passed.")
                        else:
                            retry_count += 1
                            # clear and augment again
                            step.set_augmentation_prompt("", "", "")
                            step.set_augmentation_prompt("Transformer unsound", code, ce)

                            logging.info(
                                f"Validation failed, retrying {retry_count}/{MAX_RETRIES} with augmentation..."
                            )
                    #code = result
                    success = True

            except Exception as e:
                retry_count += 1
                logging.warning(f"[STEP {index}] Exception: {e}, retrying {retry_count}/{MAX_RETRIES}")

        if not success:
            return False, "", f"[STEP {index}] Failed after {MAX_RETRIES} retries."

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
        "--log-dir", help="Path to the log directory", default="logs/", required=False
    )
    args = parser.parse_args()

    for model_name in MODEL_ENDPOINTS:
        model_out_dir = os.path.join(args.output_dir, model_name)
        if os.path.exists(model_out_dir):
            shutil.rmtree(model_out_dir)
        os.makedirs(os.path.join(model_out_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(model_out_dir, "failure"), exist_ok=True)

    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "generation.log")
    if os.path.exists(log_path):
        os.remove(log_path)
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
        for yaml_file in p.track(sorted(os.listdir(prefix))):
            full_path = os.path.join(prefix, yaml_file)
            with open(full_path, "r") as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)
            # if doc["api"] != "torch.signal.windows.nuttall":
            #    continue

            logging.info(f"{datetime.now()} - Extracting {doc['api']}")

            for model_name, url in MODEL_ENDPOINTS.items():

                model_out_dir = os.path.join(args.output_dir, model_name)
                success_dir = os.path.join(model_out_dir, "success")
                failure_dir = os.path.join(model_out_dir, "failure")
                api_name = doc["api"]

                logging.info(f"\nAPI: {api_name} -> Model: {model_name} @ {url}")
                client = TGIClient(model=url, max_new_tokens=2048)


                def generate_dsl(api, doc, dsl=None, debug=False) -> str:
                    steps = []

                    def convert(prmpt, cmpln, old) -> bool:
                        result = completion.strip()
                        return True if result.startswith("1") else False

                    steps.append(
                        Step(
                            prompter=lambda code: f"""
# DeepPoly Compatibility Check

You are a formal methods expert working on neural network verification.

Operator: Relu
Generation: 1

Operator: Absolute
Generation: 0

Oper



""",
                            composer=convert,
                            eos=["\n# END"],
                            validator=None,  # @qiuhan: add a simple validator
                        )
                    )

                    def extract_transformer_rule(prompt: str, completion: str, old_code: str) -> str:
                        # Extract a single line like: MyOp -> (...);
                        for line in completion.splitlines():
                            if "->" in line and line.strip().endswith(";"):
                                return old_code.rstrip("}") + "  " + line.strip() + "\n}"
                        # fallback if nothing parsed
                        return old_code

                    steps.append(
                        Step(
                            prompter=lambda code: f"""
# DeepPoly DSL Transformer Generation

You are a formal methods expert writing a new transformer rule for a PyTorch operator in the DeepPoly DSL. Below is the abstract domain shape and two existing examples (ReLU and Affine). Now generate a new DSL transformer for the operator below.

{DEEPPOLY_CONTEXT}

API: {api}
Documentation: {doc}
Tips:
Error Generation:
Counter Example:
Generation: (Add your transformer below. Only generate the Transformer rule (no comments, no extra output)):
""",
                            composer=extract_transformer_rule,
                            eos=["\n# END"],
                            validator=constraintflow_validator,  # @qiuhan: Constraintflow
                        )
                    )

                    steps.append(
                        Step(
                            prompter=lambda code: f"""
# DSL Transformer Tightening

You are improving the over-approximation tightness of an existing DeepPoly DSL transformer. Your goal is to reduce the gap between the upper and lower bounds, while ensuring soundness.

Context:
{DEEPPOLY_CONTEXT}

Previous transformer rule:
{code}

API: {api}
Documentation: {doc}
Tips:
Error Generation:
Counter Example:
Generation: (Now generate a tighter transformer rule that preserves soundness. Only output the new transformer rule (no comments, no explanations)):
""",
                            composer=extract_transformer_rule,
                            eos=["\n# END"],
                            validator=constraintflow_validator,  # @qiuhan: Constraintflow
                        )
                    )

                    return step_by_step_gen(client, steps)

                result, code, error = generate_dsl(
                    doc["api"], doc["doc"]
                )  
                print(code)
                if not result:
                    target_path = os.path.join(failure_dir, f"{doc['api']}.txt")
                    with open(target_path, "w") as f:
                        f.write(code)
                    logging.error(
                        f"Failed with Error:{error}\n during generating code:\n{code}\n"
                    )
                else:
                    exe, msg = executor.executor(code)
                    if exe:
                        target_path = os.path.join(success_dir, f"{doc['api']}.txt")
                        with open(target_path, "w") as f:
                            f.write(code)
                            logging.info(f"Succeed. Saved to {target_path}\n")
                    else:
                        target_path = os.path.join(failure_dir, f"{doc['api']}.txt")

                        with open(target_path, "w") as f:
                            f.write(code)
                            logging.info(
                                f"Execution Failure. Saved to {target_path}\n. Error Msg: {msg}.\n"
                            )
