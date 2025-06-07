import multiprocessing
import os
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


from vllm import LLM, SamplingParams
# login to the hugging_face with the token below in the server to access models
# TOKEN="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" # token `qiuhan_read`: hugging_face token to llama3.2; llama3.3; 


def launch_model_server(model_config, port, max_tokens=256):
    model_type = model_config["type"]
    model_id = model_config["model"]

    app = Flask(model_id)
    print(f"[✓] Starting {model_type.upper()} model on port {port}")

    if model_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
            return jsonify({
                "model": model_id,
                "prompt": prompt,
                "generated_texts": [outputs[0]["generated_text"]],
            })

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
            return jsonify({
                "model": model_id,
                "messages": messages,
                "generated_texts": [outputs[0]["generated_text"]],
            })

    elif model_type == "vllm":
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

            return jsonify({
                "model": model_id,
                "prompt": prompt,
                "generated_texts": generations,
            })

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    app.run(host="ggnds-serv-01.cs.illinois.edu", port=port)

if __name__ == "__main__":
    MODEL_PORT_PAIRS = [
        #("google/gemma-7b", 8082),
        #("meta-llama/Llama-4-Scout-17B-16E-Instruct", 8084,),  # @qiuhan: wait for access to llama4
        {"model":"meta-llama/Llama-3.3-70B-Instruct", "port": 8080,"type": "hf"},
        #{"model": "/share/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11", "port": 8080, "type": "vllm"},
        #("meta-llama/Llama-3.3-70B-Instruct", 8086),
    ]

    processes = []
    for config in MODEL_PORT_PAIRS:
        p = multiprocessing.Process(
            target=launch_model_server,
            args=(config, config["port"]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# READEME:
# use `CUDA_VISIBLE_DEVICES=3 python models.py` to run this file in a seperate terminal