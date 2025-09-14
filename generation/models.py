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

import boto3
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


def launch_model_server(
    model_config, port, host=HOST, max_tokens=256
):  # implement based on different models
    model_type = model_config["type"]  # hf, vllm, openai, aws
    model_id = model_config["model"]
    model_mode = model_config["mode"]  # chat, text

    app = Flask(model_id)
    print(f"[✓] Starting {model_id.upper()} model on port {port}")

    if any(keyword in model_id for keyword in {"gpt-4.1"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            from openai import OpenAI  # deploy through openai

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

    elif any(keyword in model_id for keyword in {"o4-mini"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            from openai import OpenAI  # deploy through openai

            client = OpenAI()

            response = client.responses.create(
                model="o4-mini",
                input=messages,
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response.output_text,
                }
            )

    elif any(keyword in model_id for keyword in {"gpt-4o"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            from openai import OpenAI  # deploy through openai

            client = OpenAI()

            response = client.responses.create(
                model="gpt-4o",
                input=messages,
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response.output_text,
                }
            )

    elif any(keyword in model_id for keyword in {"gpt-5"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            messages = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            from openai import OpenAI  # deploy through openai

            client = OpenAI()

            response = client.responses.create(
                model="gpt-5",
                input=messages,
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response.output_text,
                }
            )

    elif any(keyword in model_id for keyword in {"jamba"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            m = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "messages": m,
                    }
                ),
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": json.loads(
                        response["body"].read().decode("utf-8")
                    )["choices"][0]["message"]["content"],
                }
            )

    elif any(keyword in model_id for keyword in {"titan"}):

        @app.route("/text_generation", methods=["POST"])
        def text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1024,
                    "temperature": 0.7,
                    "topP": 0.9,
                },
            }

            response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            result = json.loads(response["body"].read())

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": result["results"][0]["outputText"],
                }
            )

    elif any(keyword in model_id for keyword in {"nova"}):

        @app.route("/text_generation", methods=["POST"])
        def text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            response = bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 1024, "temperature": 0.7, "topP": 0.9},
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response["output"]["message"]["content"][0][
                        "text"
                    ],
                }
            )

    elif any(keyword in model_id for keyword in {"claude"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            m = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            # Specifically targeted at the current project
            def convert_system_to_user_with_ack(messages):
                """
                Change the role of the first message from "system" to "user",
                and insert {"role": "assistant", "content": "okay"} right after it.
                """
                if not messages:
                    return messages

                updated_messages = messages.copy()

                if updated_messages[0].get("role") == "system":
                    updated_messages[0]["role"] = "user"

                updated_messages.insert(1, {"role": "assistant", "content": "okay"})

                return updated_messages

            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
                "messages": convert_system_to_user_with_ack(m),
            }

            response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            response_body = json.loads(response["body"].read())
            output_text = response_body["content"][0]["text"]

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": output_text,
                }
            )

    elif any(keyword in model_id for keyword in {"deepseek"}):

        @app.route("/text_generation", methods=["POST"])
        def text_generation():
            data = request.json
            prompt = data.get("question", "")
            print(prompt)
            # prompt += "\n\nReturn ONLY a single code block. No explanations, no comments." # Project specific
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            body = {"prompt": prompt, "max_tokens": 2048, "temperature": 0.7}

            resp = bedrock.invoke_model(
                modelId="us.deepseek.r1-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            data = json.loads(resp["body"].read())

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": data["choices"][0]["text"],
                }
            )

    elif any(keyword in model_id for keyword in {"llama"}):

        @app.route("/text_generation", methods=["POST"])
        def text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            body = {"prompt": prompt}

            resp = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            data = json.loads(resp["body"].read())

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": data["generation"],
                }
            )

    elif any(keyword in model_id for keyword in {"mixtral-8x7b-instruct"}):

        @app.route("/text_generation", methods=["POST"])
        def text_generation():
            data = request.json
            prompt = data.get("question", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            response = bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response["output"]["message"]["content"][0][
                        "text"
                    ],
                }
            )

    elif any(keyword in model_id for keyword in {"mistral-large-2402"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            m = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )

            response = bedrock.invoke_model(
                modelId=model_id, body=json.dumps({"messages": m})
            )

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": json.loads(
                        response["body"].read().decode("utf-8")
                    )["choices"][0]["message"]["content"],
                }
            )

    elif any(keyword in model_id for keyword in {"gpt-oss"}):

        @app.route("/chat", methods=["POST"])
        def text_generation():
            data = request.json
            m = data.get("messages", "")
            temperature = float(data.get("temperature", 1.0))
            max_new_tokens = int(data.get("max_tokens", max_tokens))

            system_msg = data.get("system_msg", None)

            bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

            native_request = {
                "messages": m,
                "max_completion_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False,  # You can omit this field
            }

            # Make the InvokeModel request
            response = bedrock.invoke_model(
                modelId=model_id, body=json.dumps(native_request)
            )

            response = json.loads(response["body"].read().decode("utf-8"))

            return jsonify(
                {
                    "model": model_id,
                    "generated_texts": response["choices"][0]["message"]["content"],
                }
            )

    else:
        raise ValueError(f"Unknown model name: {model_id}")

    app.run(host=host, port=port)


"""
    elif any(
        keyword in model_id
        for keyword in {
            "deepseek",
        }
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        llm = LLM(
            model=model_id, # run locally
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

    elif any(
        keyword in model_id.lower()
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
        def text_generation():
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
        def text_generation():
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
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch one or more model servers.")

    parser.add_argument(
        "--model",
        "-m",
        nargs="+",  # multiple models
        default=["deepseek"],
        choices=[
            "llama",
            "llama3.3",
            "llama4",
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
        help=(
            "One or more model keywords to launch. E.g., --model deepseek llama. "
            "Available options: llama-3.3, llama-4, deepseek, gpt-4o, gpt-4.1, o4-mini"
            "Default is ['deepseek']."
        ),
    )

    parser.add_argument(
        "--host",
        "-host",
        type=str,
        default=HOST,
        help="Host IP to bind Flask servers",
    )

    args = parser.parse_args()

    if isinstance(args.model, str):
        requested_models = [args.model.lower()]
    else:
        requested_models = [m.lower() for m in args.model]
    # requested_models = [m.lower() for m in args.model]

    selected_configs = []
    for keyword in requested_models:
        for config in MODEL_PORT_PAIRS:
            # if keyword == "aws" and config["type"] == "aws": # to call all the models through aws bedrock
            #    selected_configs.append(config)
            if (
                keyword in config["model"].lower()
            ):  # to call gpt-4.1, gpt-4o, o4-mini, gpt-5
                selected_configs.append(config)
                # break

    if not selected_configs:
        raise ValueError(f"No models matched: {requested_models}")

    processes = []
    for config in selected_configs:
        p = multiprocessing.Process(
            target=launch_model_server,
            args=(config, config["port"], args.host),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# READEME:
# use `python models.py -m [gpt-4o]/[]..` to run this file in a seperate terminal --> multiple model in multiple terminals
