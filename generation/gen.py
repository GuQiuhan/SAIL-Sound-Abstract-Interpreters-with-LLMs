"""
## Step-by-step Prompting for Long DSL Generation
0. Let output spec code S = "class "
1. Decompose the DSL generation task into multiple steps. (TBD)
2. In each step, use a step-specific prompt comprised of {few-shot examples, constraints, and instructions}.
3. Parse the prompt+output to S, continue to a new step of 2.
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
    "deepseek-v2-lite": "http://ggnds-serv-01.cs.illinois.edu:8080",
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

opt_list = [
    "Abs",
    "Affine",
    "Avgpool",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "Maxpool",
    "Minpool",
    "Neuron_add",
    "Neuron_list_mult",
    "Neuron_max",
    "Neuron_min",
    "Neuron_mult",
    "Relu",
    "Relu6",
    "rev_Abs",
    "rev_Affine",
    "rev_HardSigmoid",
    "rev_HardSwish",
    "rev_HardTanh",
    "rev_Maxpool",
    "rev_Neuron_add",
    "rev_Neuron_max",
    "rev_Neuron_min",
    "rev_Neuron_mult",
    "rev_Relu",
    "rev_Relu6"
]

class Step:
    def __init__(self, prompter, composer=None, eos=None, validator=None):
        self.prompter: Callable[[str, Optional[str]], str] = prompter
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[str, str, str], Union[str, bool]] = composer
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


def step_by_step_gen(client: Client, steps: List[Step]):
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
        new_code =""
        while retry_count < MAX_RETRIES and not success:
            
            code = ""
            prompt = step.prompter(code)
            prompt = step.prompter_with_augmentation(prompt)

            completion = client.textgen(prompt=prompt)


            print("here")
            print(completion)

            if "Model Generation Error" in completion:
                return False, "", f"[STEP {index}] Model Generation Error during completion."

            new_code = step.composer(prompt, completion, code)
            print(f"--- STEP {index} COMPLETION ---\n{completion}\n")
            print(f"--- STEP {index} PARSED RESULT ---\n{new_code}\n")

            # here we just have one step
            if step.validator:
                result, ce = step.validator(new_code)
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
            else:
                success = True



        if not success:
            return False, code, f"[STEP {index}] Failed after {MAX_RETRIES} retries."

    return True, new_code, ""



        
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
    log_path = os.path.join(args.log_dir, "deepseek_generation.log")
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
        for op_name in p.track(sorted(opt_list)):
            doc = {"api": op_name}

            logging.info(f"{datetime.now()} - Extracting {doc['api']}")

            for model_name, url in MODEL_ENDPOINTS.items():
                model_out_dir = os.path.join(args.output_dir, model_name)
                success_dir = os.path.join(model_out_dir, "success")
                failure_dir = os.path.join(model_out_dir, "failure")
                api_name = doc["api"]

                logging.info(f"\nAPI: {api_name} -> Model: {model_name} @ {url}")
                client = TGIClient(model=url, max_new_tokens=2048)


                def generate_dsl(api, dsl=None, debug=False) -> str:
                    steps = []

                    def extract_constraintflow_block(prmpt, cmpl, code) -> str:
                        """
                        Extract everything starting from the 'deeppoly' keyword until the closing brace '}' that balances the opening one.
                        """
                        match = re.search(r'(deeppoly\s*\{)', cmpl)
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

                    steps.append(
                        Step(
                            prompter=lambda code: f"""
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepPoly transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

{CONSTRAINTFLOW}

### Example: ReLU operator
Input: Generate the transformer for `relu` operator
Output:
{prmpt_relu}

### Example: Abs operator
Input: Generate the transformer for `abs` operator
Output:
{prmpt_abs}

### Now generate the transformer for {api} operator
Input: Generate the transformer for {api} operator
Output:
""",
                            composer=extract_constraintflow_block,
                            eos=["\n# END"],
                            #validator= constraintflow_validator,  # @qiuhan: add a simple validator
                            validator=None,
                        )
                    )

                    return step_by_step_gen(client, steps)

                result, code, error = generate_dsl(doc["api"])  

                

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
