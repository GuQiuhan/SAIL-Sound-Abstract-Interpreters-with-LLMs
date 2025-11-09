import json
import traceback
from abc import ABC, abstractmethod

import openai
import requests
from huggingface_hub import InferenceClient

from generation.utils import *


class Client(ABC):
    def __init__(self, model, max_new_tokens=2048, temperature=1.0, do_sample=False):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    @abstractmethod
    def textgen(self, prompt) -> str:
        pass


# test generation
class TGIClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = InferenceClient(model=self.model)

    def textgen(self, prompt, **kwargs) -> str:
        # def textgen(self, question, n=1, max_tokens=4096, temperature=0.0, system_msg=None):
        url = f"{self.model}/text_generation"
        headers = {"Content-Type": "application/json"}
        system_msg = "You are a highly skilled software engineer. Your task is to generate high-quality, efficient, and well-commented code based on the user's prompt. The user will provide a prompt describing a task or functionality, and you should respond with the appropriate code in the specified programming language. Include only the code in your response, and avoid any unnecessary explanations unless explicitly requested. Use best practices for coding, including meaningful variable names, comments, and proper formatting."
        data = {
            "question": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "system_msg": "",
        }

        # response = requests.post(url, headers=headers, json=data)

        # response.raise_for_status()
        # return response.json().get("generated_texts", [])[0].strip('[]')

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json().get("generated_texts", [[]])
            # return response.json().get("generated_texts", [])[-1]["content"].strip("[]")
        except Exception as e:
            return f"Model Generation Error: {type(e).__name__}"

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        messages: list of dicts, each with 'role' and 'content'
        """
        url = f"{self.model}/chat"
        headers = {"Content-Type": "application/json"}
        system_msg = "You are a highly skilled software engineer. Your task is to generate high-quality, efficient, and well-commented code based on the user's prompt. The user will provide a prompt describing a task or functionality, and you should respond with the appropriate code in the specified programming language. Include only the code in your response, and avoid any unnecessary explanations unless explicitly requested. Use best practices for coding, including meaningful variable names, comments, and proper formatting."
        data = {
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "system_msg": "",
        }

        # response = requests.post(url, headers=headers, json=data)

        # response.raise_for_status()
        # return response.json().get("generated_texts", [])[0].strip('[]')

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            messages = response.json().get("generated_texts", [[]])

            if isinstance(messages, str):
                return messages

            messages = messages[0]
            last_reply = next(
                (
                    msg["content"]
                    for msg in reversed(messages)
                    if msg.get("role") == "assistant"
                ),
                "",
            )

            return last_reply

        except Exception as e:
            traceback.print_exc()
            return f"Model Generation Error: {type(e).__name__}"


if __name__ == "__main__":
    client = TGIClient(model="http://ggnds-serv-01.cs.illinois.edu:6046")

    cex = """
# You can modify previously generated (invalid) code to make it sound:

transformer deeppoly{
    HardSwish ->
        ((prev[l] >= 3) ? (prev[l], prev[u], prev, prev) :
         ((prev[u] <= -3) ? (0, 0, 0, 0) :
          ((prev[l] >= -3 && prev[u] <= 3) ?
           (min(f1(prev[l]), f1(prev[u])), max(f1(prev[l]), f1(prev[u])),
            f2(prev) * (prev >= -3 && prev <= 3),
            f2(prev) * (prev >= -3 && prev <= 3)) :
           (((prev[l] < -3 && prev[u] > -3 && prev[u] <= 3) ?
             (0, f1(prev[u]), 0, f2(prev) * (prev >= -3 && prev <= 3)) :
             (((prev[l] >= -3 && prev[u] > 3) ?
               (f1(prev[l]), prev[u], f2(prev) * (prev >= -3 && prev <= 3), prev) :
               (((prev[l] < -3 && prev[u] > 3) ?
                 (0, prev[u], 0, prev) :
                 (0, 0, 0, 0))))))));
}

Counterexample: Incorrect when prev[l]<-1.5 <prev[u]


# Learn from the unsound generation above and revise your output accordingly. Output the DSL only."

"""

    op_appen = op_appendix.get("HardSwish", "")

    message1 = [
        {
            "role": "system",
            "content": f"{DEEPPOLY_CONSTRAINTFLOW}",
        },
        {"role": "user", "content": "Generate the transformer for `relu` operator "},
        {"role": "assistant", "content": f"{prmpt_relu_deeppoly}"},
        {"role": "user", "content": "Generate the transformer for `abs` operator "},
        {"role": "assistant", "content": f"{prmpt_abs_deeppoly}"},
        {"role": "user", "content": "Generate the transformer for `affine` operator "},
        {"role": "assistant", "content": f"{prmpt_affine_deeppoly}"},
        {
            "role": "user",
            "content": f"Generate the transformer for `HardSwish` operator. {op_appen}",
        },
    ]

    prompt1 = f"""
    {DEEPPOLY_CONSTRAINTFLOW}

    ### Example: ReLU operator
    Input: Generate the transformer for `relu` operator
    Output:
    {prmpt_relu_deeppoly}

    ### Example: Abs operator
    Input: Generate the transformer for `abs` operator
    Output:
    {prmpt_abs_deeppoly}

    ### Example: Affine operator
    Input: Generate the transformer for `affine` operator
    Output:
    {prmpt_affine_deeppoly}

    ### Now generate the transformer for `HardSwish` operator with transition points -3 and 3
    Input: Generate the transformer for `HardSwish` operator. {op_appen} {cex}
    Output:


    """

    # try:
    #    output = client.chat(messages=message1)

    # except:
    #    try:
    output = client.textgen(prompt=prompt1)
    #    except:
    #        print("Wrong model API call.")

    print(output)
