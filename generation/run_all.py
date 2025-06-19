"""
Script: run_all.py
------------------

This script automates the process of:
1. Starting the local model server (e.g., DeepSeek, GPT-4, LLaMA) by running models.py
2. Generating DSL transformers via step-by-step prompting using gen.py
3. Cleaning up by terminating the model server after generation

Usage:
    python run_all.py --model deepseek --certifier deeppoly

Arguments:
    --model (-m):     Specifies which model to launch (e.g., deepseek, gpt-4o, llama-3.3)
                      Default is "deepseek".

    --certifier (-c): Specifies which abstract domain to use (e.g., deeppoly, ibp, deepz)
                      Default is "deeppoly".

Features:
- Automatically starts the selected model server.
- Waits briefly for the model to initialize.
- Runs gen.py to generate and validate DSL transformer code.
- Terminates the model process after generation is complete.
- Colored terminal output for status messages.

Note:
- `models.py` must expose a model server on a known port based on model name.
- `gen.py` must accept the same --model and --certifier arguments.

"""


import argparse
import os
import signal
import subprocess
import time

parser = argparse.ArgumentParser(description="Run model.py and gen.py in sequence.")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="deepseek",
    choices=["llama-3.3", "llama-4", "deepseek", "gpt-4o", "gpt-4.1", "o4-mini"],
    help="Name of the model to run. Options: llama3, deepseek. " "Default is deepseek.",
)

parser.add_argument(
    "-c",
    "--certifier",
    type=str,
    default="deeppoly",
    choices=["deeppoly", "ibp", "deepz"],
    help="Name of the certifier to use. Options: deeppoly, ibp, deepz. "
    "Default is deeppoly.",
)
args = parser.parse_args()

model_process = subprocess.Popen(
    ["python", "models.py", "--model", args.model],
    # stdout=subprocess.DEVNULL,
    # stderr=subprocess.DEVNULL
)

print(
    f"\033[94mStarted model.py with model {args.model} (PID: {model_process.pid})\033[0m"
)


time.sleep(5)

print(
    f"\033[93mRunning gen.py with model {args.model} and certifier {args.certifier}...\033[0m"
)

gen_exit_code = subprocess.call(
    ["python", "gen.py", "--model", args.model, "--certifier", args.certifier]
)

print("\033[92mGeneration finished.\033[0m Terminating model.py...")

os.kill(model_process.pid, signal.SIGTERM)

print("\033[91mDone.\033[0m")
