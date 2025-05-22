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

# set max retry time every turn
MAX_RETRIES = 10

MODEL_ENDPOINTS = {
    #"gemma-7b": "http://10.192.122.120:8082",
    #"deepseek-6.7b": "http://10.192.122.120:8083",
    "llama3-1B": "http://ggnds-serv-01.cs.illinois.edu:8080",
    #"llama3-70B": "http://10.192.122.120:8086",
}

DEEPPOLY_CONTEXT = """
Def shape as (Real l, Real u, PolyExp L, PolyExp U) {
    [curr[l] <= curr, curr[u] >= curr, curr[L] <= curr, curr[U] >= curr]
};

Func concretize_lower(Neuron n, Real c) = (c >= 0) ? (c * n[l]) : (c * n[u]);
Func concretize_upper(Neuron n, Real c) = (c >= 0) ? (c * n[u]) : (c * n[l]);

Func replace_lower(Neuron n, Real c) = (c >= 0) ? (c * n[L]) : (c * n[U]);
Func replace_upper(Neuron n, Real c) = (c >= 0) ? (c * n[U]) : (c * n[L]);

Func priority(Neuron n) = n[layer];

Func backsubs_lower(PolyExp e, Neuron n) = 
    (e.traverse(backward, priority, false, replace_lower){e <= n}).map(concretize_lower);

Func backsubs_upper(PolyExp e, Neuron n) = 
    (e.traverse(backward, priority, false, replace_upper){e >= n}).map(concretize_upper);

Transformer DeepPoly(curr, prev){
ReLU -> prev[l] > 0 ? (prev[l], prev[u], prev, prev) :
         (prev[u] < 0 ? (0, 0, 0, 0) :
         (0, prev[u], 0, ((prev[u] / (prev[u] - prev[l])) * prev) - ((prev[u] * prev[l]) / (prev[u] - prev[l]))));

Affine -> (
    backsubs_lower(prev.dot(curr[w]) + curr[b], curr),
    backsubs_upper(prev.dot(curr[w]) + curr[b], curr),
    prev.dot(curr[w]) + curr[b],
    prev.dot(curr[w]) + curr[b]
);
"""



class Step:
    def __init__(self, prompter, composer=None, eos=None, validator=None):
        self.prompter: Callable[[str, Optional[str]], str] = prompter
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[str, str, str], Union[str, bool]] = composer
        self.validator: Callable[[str], None] = validator  # validate the code
        # add augmentation prompting
        self.aug_prompt = ""
        self.error_examples = ""  # List to store error examples
        self.given_code = ""  # provided code

    def set_augmentation_prompt(self, aug_prompt: str, error_example: str):
        self.aug_prompt = aug_prompt
        self.error_examples = error_example

    def prompter_with_augmentation(self, old_prmpt: str) -> str:
        """Generates the prompt with augmentation and error examples."""
        augmented_prompt = old_prmpt
        if self.aug_prompt:
            last_api_index = augmented_prompt.rfind("API: ")
            error_index = augmented_prompt.find("Error Example:", last_api_index)
            if error_index != -1:
                augmented_prompt = (
                    augmented_prompt[:error_index]
                    + "\n"
                    + self.aug_prompt
                    + "\n"
                    + augmented_prompt[error_index:]
                )
            last_api_index = augmented_prompt.rfind("API: ")
            generation_index = augmented_prompt.find("Class Context:", last_api_index)
            if generation_index != -1:
                augmented_prompt = (
                    augmented_prompt[:generation_index]
                    + "\n```python\n"
                    + self.error_examples
                    + "\n```\n"
                    + augmented_prompt[generation_index:]
                )

        return augmented_prompt


def step_by_step_gen(client: Client, steps: List[Step]):
    """
    Executes a sequence of steps for DSL generation.
    Step 1: DeepPoly support check (returns bool)
    Step 2: DSL transformer generation (returns DSL code if applicable)

    Returns:
        (success: bool, result: str, error_message: str)
    """
    code = ""
    for index, step in enumerate(steps, start=1):
        logging.info(f"[STEP {index}] Starting step {index}/{len(steps)}")
        retry_count = 0
        success = False

        while retry_count < MAX_RETRIES and not success:
            try:
                prompt = step.prompter(code)
                prompt = step.prompter_with_augmentation(prompt)

                print(prompt)

                completion = client.textgen(prompt=prompt, stop_sequences=step.eos)

                print(completion)

                if "Model Generation Error" in completion:
                    return False, "", f"[STEP {index}] Model Generation Error during completion."

                result = step.composer(prompt, completion, code)
                print(f"--- STEP {index} COMPLETION ---\n{completion}\n")
                print(f"--- STEP {index} PARSED RESULT ---\n{result}\n")

                # Step 1 returns bool
                if index == 1 and isinstance(result, bool):
                    if result:
                        logging.info("[STEP 1] Operator is supported by DeepPoly.")
                        success = True
                        code = ""  # reset for step 2
                    else:
                        return False, "", "[STEP 1] Operator is not supported by DeepPoly. Skipping transformer generation."
                else:
                    # Step 2 or others: update code directly
                    code = result
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

You are a formal methods expert working on neural network verification. Your task is to determine whether the following PyTorch operator is compatible with the DeepPoly abstract domain.

DeepPoly supports any operator for which:
- The output can be overapproximated using affine bounds (symbolic linear expressions).
- The operator is monotonic or piecewise-linear (e.g., ReLU, LeakyReLU, HardTanh).
- The operator can be decomposed into affine and element-wise operations (e.g., Add, Mul, Clamp).
- The operator acts in an element-wise or structured way (e.g., pooling, affine transforms).

DeepPoly does NOT support:
- Operators with non-elementwise behavior that cannot be soundly approximated with affine bounds.
- Non-monotonic or highly nonlinear operators like Softmax, Argmax, Dropout, Sampling, or control flow.

API: {api}
****************************
Documentation: {doc}
****************************

Return:
- `1` if the operator is supported.
- `0` if the operator is not supported.
""",
                            composer=convert,
                            eos=["\n# END"],
                            validator=None,  # @qiuhan: Constraintflow
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

Add your transformer below. Only generate the Transformer rule (no comments, no extra output):
""",
                            composer=extract_transformer_rule,
                            eos=["\n# END"],
                            validator=None,  # @qiuhan: Constraintflow
                        )
                    )

                    return step_by_step_gen(client, steps)

                result, code, error = generate_dsl(
                    doc["api"], doc["doc"]
                )  
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
