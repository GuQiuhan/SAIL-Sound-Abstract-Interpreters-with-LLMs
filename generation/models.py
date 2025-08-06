"""
Script: models.py
------------------

This script launches one **or more** model servers locally using Flask, serving as backends for text generation and chat interfaces.

Supported Models:
    - deepseek (via vLLM)
    - llama-3.3 / llama-4 (via Hugging Face Transformers)
    - gpt-4o / gpt-4.1 / o4-mini (via OpenAI API)

Usage Example:
    python models.py --model deepseek
    CUDA_VISIBLE_DEVICES=0,1 python models.py --model llama-4 gpt-4o

Note:
    - Make sure `utils.py` defines `MODEL_PORT_PAIRS`
    - This script is designed to work with a companion script (e.g., gen.py) that queries the endpoints.
"""


import argparse
import json
import multiprocessing
import os

import openai
import requests
import torch
from flask import Flask, jsonify, request

# use your own openai key here
from key import OPENAI_KEY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import *

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from vllm import LLM, SamplingParams

# login to the hugging_face with the token below in the server to access models
# TOKEN="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" # token `qiuhan_read`: hugging_face token to llama3.2; llama3.3;


def launch_model_server(model_config, port, max_tokens=256):
    model_type = model_config["type"]
    model_id = model_config["model"]

    app = Flask(model_id)
    print(f"[✓] Starting {model_id.upper()} model on port {port}")

    if any(keyword in model_id for keyword in {"o4-mini", "gpt-4.1", "gpt-4o"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            from openai import OpenAI

            client = OpenAI()

            response = client.responses.create(
                model="gpt-4.1",
                input=messages,
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response.output_text,
                }
            )

    elif any(
        keyword in model_id
        for keyword in {
            "llama-3.3",
            "llama-4",
        }
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("✅ huggingface model loaded.")

        @app.route("/text_generation", methods=["POST"])
        def hf_text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            outputs = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": [outputs[0]["generated_text"]],
                }
            )

        @app.route("/chat", methods=["POST"])
        def hf_chat():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": [outputs[0]["generated_text"]],
                }
            )

    elif any(
        keyword in model_id
        for keyword in {
            "deepseek",
        }
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,
        )
        print("✅ vLLM Model loaded.")

        @app.route("/text_generation", methods=["POST"])
        def vllm_text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 0.6))
            top_p = float(data.get("top_p", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )

            outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
            generations = [output.outputs[0].text for output in outputs]

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": generations,
                }
            )

    else:
        raise ValueError(f"Unknown model name: {model_id}")

    app.run(host="ggnds-serv-01.cs.illinois.edu", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch one or more model servers.")

    parser.add_argument(
        "--model",
        "-m",
        nargs="+",  # multiple models
        default=["deepseek"],
        choices=["llama-3.3", "llama-4", "deepseek", "gpt-4o", "gpt-4.1", "o4-mini"],
        help=(
            "One or more model keywords to launch. E.g., --model deepseek llama. "
            "Available options: llama-3.3, llama-4, deepseek, gpt-4o, gpt-4.1, o4-mini"
            "Default is ['deepseek']."
        ),
    )

    args = parser.parse_args()
    requested_models = [m.lower() for m in args.model]

    selected_configs = []
    for config in MODEL_PORT_PAIRS:
        for keyword in requested_models:
            if keyword in config["model"].lower():
                selected_configs.append(config)
                break

    if not selected_configs:
        raise ValueError(f"No models matched: {requested_models}")

    processes = []
    for config in selected_configs:
        p = multiprocessing.Process(
            target=launch_model_server,
            args=(config, config["port"]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# READEME:
# use `python models.py -m [gpt-4o]/[]..` to run this file in a seperate terminal
