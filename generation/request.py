import json
from abc import ABC, abstractmethod

import openai
import requests
from huggingface_hub import InferenceClient


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
            return response.json().get("generated_texts", [])[0].strip("[]")
            #return response.json().get("generated_texts", [])[-1]["content"].strip("[]")
        except Exception as e:
            return f"Model Generation Error: {type(e).__name__}"


if __name__ == "__main__":
    client = TGIClient(model="http://ggnds-serv-01.cs.illinois.edu:8080")

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

    output = client.textgen(prompt = f"""
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepPoly transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

{CONSTRAINTFLOW}

### Example: ReLU operator
Input: Generate the transformer for `relu` operator
Output:
```dsl
{prmpt_relu}
```

### Example: Abs operator
Input: Generate the transformer for `abs` operator
Output:
```dsl
{prmpt_abs}
```

### Now generate the transformer for `relu6` operator
Input: Generate the transformer for `relu6` operator
Output:
"""
    )
    print(output)


