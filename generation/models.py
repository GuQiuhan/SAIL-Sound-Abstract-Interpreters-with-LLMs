import multiprocessing

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# login to the hugging_face with the token below in the server to access models
# TOKEN="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" # token `qiuhan_read`: hugging_face token to llama3.2; llama3.3; 


def launch_model_server(model_id: str, port: int, max_new_tokens: int = 256):
    print(f"[✓] Starting server for {model_id} on port {port}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    app = Flask(model_id)

    @app.route("/text_generation", methods=["POST"])
    def text_generation():
        data = request.json
        prompt = data.get("question", "")
        max_tokens = data.get("max_tokens", max_new_tokens)
        temperature = data.get("temperature", 1.0)

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        return jsonify(
            {
                "model": model_id,
                "prompt": prompt,
                "generated_texts": [outputs[0]["generated_text"]],
            }
        )

    app.run(host="ggnds-serv-01.cs.illinois.edu", port=port)  # @qiuhan: change the host here if needed


if __name__ == "__main__":
    MODEL_PORT_PAIRS = [
        #("google/gemma-7b", 8082),
        #("deepseek-ai/deepseek-coder-6.7b-instruct", 8083,),  # @qiuhan: test deepseek again
        #("meta-llama/Llama-4-Scout-17B-16E-Instruct", 8084,),  # @qiuhan: wait for access to llama4
        ("meta-llama/Llama-3.2-1B-Instruct", 8080),
        #("meta-llama/Llama-3.3-70B-Instruct", 8086),
    ]

    processes = []
    for model_id, port in MODEL_PORT_PAIRS:
        p = multiprocessing.Process(target=launch_model_server, args=(model_id, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# READEME:
# use `CUDA_VISIBLE_DEVICES=3 python models.py` to run this file in a seperate terminal