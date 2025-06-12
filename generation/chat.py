"""
## Step-by-step Prompting for Long DSL Generation
0. Let output spec code S = "class "
1. Decompose the DSL generation task into multiple steps. (TBD)
2. In each step, use a step-specific prompt comprised of {few-shot examples, constraints, and instructions}.
3. Parse the prompt+output to S, continue to a new step of 2.
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

from validator.validate_dsl import constraintflow_validator

# set max retry time every turn
MAX_RETRIES = 10

MODEL_ENDPOINTS = {
    #"Llama-3.3": "http://ggnds-serv-01.cs.illinois.edu:8080",
    #"GPT-4.1": "http://ggnds-serv-01.cs.illinois.edu:8080",
    "GPT-4o": "http://ggnds-serv-01.cs.illinois.edu:8080",
}


CONSTRAINTFLOW = """
DeepPoly certifier uses four kinds of bounds to approximate the operator: (Float l, Float u, PolyExp L, PolyExp U).
They must follow the constraints that: curr[l] <= curr <= curr[u] & curr[L] <= curr <= curr[U]. `curr` here means the current neuron, `prev` means the inputs to the operator.
When the operator takes multiple inputs, use `prev_0`, `prev_1`, ... to refer to each input.  
So every transformer in each case of the case analysis must return four values. Use any funstions below if needed instead of use arithmetic operators.
Function you can use:
- func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
- func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);
- func replace_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[L]) : (coeff * n[U]);
- func replace_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[U]) : (coeff * n[L]);
- func priority(Neuron n) = n[layer];
- func priority2(Neuron n) = -n[layer];
- func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, true, replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, true, replace_upper){e >= n}).map(simplify_upper);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
Don't add comments to DSL.
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

prmpt_affine = """
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Affine -> (backsubs_lower(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), backsubs_upper(prev.dot(curr[weight]) + curr[bias], curr, curr[layer]), prev.dot(curr[weight]) + curr[bias], prev.dot(curr[weight]) + curr[bias]);
}
"""

opt_list = [
    "Abs",
    "Add",
    "Affine",
    "Avgpool",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "Max",
    "Maxpool",
    "Min",
    "Minpool",
    "Mult",
    "Relu",
    "Relu6",
]

class Step:
    def __init__(self, prompter, composer=None, eos=None, validator=None):
        self.prompter: Callable[[Optional[str]], List[Dict[str, str]]] = prompter # for chat models
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
        while retry_count < MAX_RETRIES and not success:
            
            code = ""
            message = step.prompter(code)
            #prompt = step.prompter_with_augmentation(prompt)

            #completion = client.textgen(prompt=prompt)
            completions = [client.chat(messages=message) for _ in range(3)] # multiple(3) samples

            for sample_id, completion in enumerate(completions, start=1):
                if "Model Generation Error" in completion:
                    logging.warning(f"[STEP {index}] Sample {sample_id}: Model Generation Error")
                    continue

                code = step.composer("", completion, code)
                print(f"[STEP {index}] Sample {sample_id}: Completion:\n{completion}")
                print(f"[STEP {index}] Sample {sample_id}: Parsed DSL:\n{code}")

                if step.validator:
                    try:
                        result, ce = step.validator(code)
                    except Exception as e:
                        #logging.warning(f"[STEP {index}] Sample {sample_id}: Validator exception: {e}. Code: \n {code}\n")
                        logging.warning(f"[STEP {index}] Sample {sample_id}: Validator exception. Full traceback:\n{traceback.format_exc()}\nCode:\n{code}\n")

                        result, ce = False, None

                    if result:
                        success = True
                        logging.info(f"[STEP {index}] Sample {sample_id}: Validation passed.")
                        break
                    else:
                        logging.info(f"[STEP {index}] Sample {sample_id}: Validation failed.")
                else:
                    success = True
                    break

            if not success:
                retry_count += 1
                logging.info(
                    f"[STEP {index}] All {len(completions)} samples failed validation. Retrying {retry_count}/{MAX_RETRIES}..."
                )


        if not success:
            return False, code, f"[STEP {index}] Failed after {MAX_RETRIES} retries."

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

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", run_timestamp)
    result_dir = os.path.join(run_dir, "results")
    log_path = os.path.join(run_dir, "generation.log")

    for model_name in MODEL_ENDPOINTS:
        model_out_dir = os.path.join(result_dir, model_name)
        if os.path.exists(model_out_dir):
            shutil.rmtree(model_out_dir)
        os.makedirs(os.path.join(model_out_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(model_out_dir, "failure"), exist_ok=True)



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

        for op_name in p.track(sorted(opt_list)):
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
                    
                    def prmpt(code: Optional[str]) -> List[dict]:
                        return [
    {"role": "system", "content": f"You are a formal methods expert working on neural network verification. Your task is to generate the DeepPoly transformers for DNN operators. Generate the transformer in Constraintflow DSL. {CONSTRAINTFLOW}"},
    {"role": "user", "content": "Generate the transformer for `relu` operator "},
    {"role": "assistant", "content": prmpt_relu},
    {"role": "user", "content": "Generate the transformer for `abs` operator "},
    {"role": "assistant", "content": prmpt_abs},
    {"role": "user", "content": "Generate the transformer for `affine` operator "},
    {"role": "assistant", "content": prmpt_affine},
    {"role": "user", "content": f"Generate the transformer for {api} operator "},
                        ]


                    steps.append(
                        Step(
                            prompter=prmpt,
                            composer=extract_constraintflow_block,
                            eos=["\n# END"],
                            validator= constraintflow_validator,  # @qiuhan: add a simple validator
                            #validator=None,
                        )
                    )

                    return step_by_step_gen(client, steps)

                result, code, error = generate_dsl(doc["api"])  
                op_end_time = time()  # 每个 operator 结束时间
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