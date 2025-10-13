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
    client = TGIClient(model="http://ggnds-serv-01.cs.illinois.edu:8086")

    cex = """
# Previously generated (invalid) code:

def Shape as (Float l, Float u){[(curr[l]<=curr),(curr[u]>=curr)]};

transformer ibp{
    Gelu -> (0.5*prev[l]*(1+erf(prev[l]/1.4142135623730951)), 0.5*prev[u]*(1+erf(prev[u]/1.4142135623730951)));
}

# Counter Example respectively: [prev_l,prev_u]=[-2,-0.2]
# Learn from the failed generation above and revise your output accordingly. Output the DSL only."

"""
    CONSTRAINTFLOW = """
You are a formal methods expert working on neural network verification.
Your task is to generate the IBP transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

Here is the grammar of Constraintflow DSL:

'''
expr_list : expr COMMA expr_list
    |   expr ;

exprs: expr exprs
    | expr;

metadata: WEIGHT
    |   BIAS
    |   EQUATIONS
    |   LAYER ;

expr: FALSE                                         #false
    | TRUE                                          #true
    | IntConst                                      #int
    | FloatConst                                    #float
    | VAR                                           #varExp
    | EPSILON                                       #epsilon
    | CURR                                          #curr
    | PREV                                          #prev
    | PREV_0                                        #prev_0
    | PREV_1                                        #prev_1
    | CURRLIST                                      #curr_list
    | LPAREN expr RPAREN                            #parenExp
    | LSQR expr_list RSQR                           #exprarray
    | expr LSQR metadata RSQR                       #getMetadata
    | expr LSQR VAR RSQR                            #getElement
    | expr binop expr                               #binopExp
    | NOT expr                                      #not
    | MINUS expr                                    #neg
    | expr QUES expr COLON expr                     #cond
    | expr DOT TRAV LPAREN direction COMMA expr COMMA expr COMMA expr RPAREN LBRACE expr RBRACE     #traverse
    | argmax_op LPAREN expr COMMA expr RPAREN       #argmaxOp
    | max_op LPAREN expr RPAREN                     #maxOpList
    | max_op LPAREN expr COMMA expr RPAREN          #maxOp
    | list_op LPAREN expr RPAREN                    #listOp
    | expr DOT MAP LPAREN expr RPAREN               #map
    | expr DOT MAPLIST LPAREN expr RPAREN           #map_list
    | expr DOT DOTT LPAREN expr RPAREN              #dot
    | expr DOT CONCAT LPAREN expr RPAREN            #concat
    | LP LPAREN lp_op COMMA expr COMMA expr RPAREN  #lp
    | VAR LPAREN expr_list RPAREN                   #funcCall
    | VAR exprs                                     #curry
;

trans_ret :
    expr QUES trans_ret COLON trans_ret #condtrans
    | LPAREN trans_ret RPAREN #parentrans
    | expr_list #trans
;
'''

IBP certifier uses two kinds of bounds to overapproximate the operator: (Float l, Float u).
They must follow the constraints that: curr[l] <= curr <= curr[u]. `curr` here means the current neuron, `prev` means the inputs to the operator.
When the operator takes multiple inputs, use `prev_0`, `prev_1`, ... to refer to each input.
So every transformer in each case of the case analysis must return two values. Use any functions below if needed instead of using arithmetic operators.

Functions you can use:
- func simplify_lower(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[l]) : (coeff * n[u]);
- func simplify_upper(Neuron n, Float coeff) = (coeff >= 0) ? (coeff * n[u]) : (coeff * n[l]);
- func abs(Float x) = x > 0 ? x : -x;
- func max_lower(Neuron n1, Neuron n2) = n1[l]>=n2[l] ? n1[l] : n2[l];
- func max_upper(Neuron n1, Neuron n2) = n1[u]>=n2[u] ? n1[u] : n2[u];
- func min_lower(Neuron n1, Neuron n2) = n1[l]<=n2[l] ? n1[l] : n2[l];
- func min_upper(Neuron n1, Neuron n2) = n1[u]<=n2[u] ? n1[u] : n2[u];
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func sigma(Float x) = 1/(1+eps(-x));
- func erf(Float x)
- func priority(Neuron n) = n[layer];

Don't add comments to DSL.
"""

    message1 = [
        {
            "role": "system",
            "content": f"{CONSTRAINTFLOW}",
        },
        {"role": "user", "content": "Generate the transformer for `relu` operator "},
        {"role": "assistant", "content": f"{prmpt_relu_ibp}"},
        {"role": "user", "content": "Generate the transformer for `abs` operator "},
        {"role": "assistant", "content": f"{prmpt_abs_ibp}"},
        {"role": "user", "content": "Generate the transformer for `affine` operator "},
        {"role": "assistant", "content": f"{prmpt_affine_ibp}"},
        {
            "role": "user",
            "content": f"Generate the transformer for `gelu` operator. {cex}",
        },
    ]

    prompt1 = f"""
    {IBP_CONSTRAINTFLOW}

    ### Example: ReLU operator
    Input: Generate the transformer for `relu` operator
    Output:
    {prmpt_relu_ibp}

    ### Example: Abs operator
    Input: Generate the transformer for `abs` operator
    Output:
    {prmpt_abs_ibp}

    ### Example: Affine operator
    Input: Generate the transformer for `affine` operator
    Output:
    {prmpt_affine_ibp}

    ### Now generate the transformer for `Gelu` operator
    Input: Generate the transformer for `Gelu` operator
    Output:
    """

    try:
        output = client.chat(messages=message1)

    except:
        try:
            output = client.textgen(prompt=prompt1)
        except:
            print("Wrong model API call.")

    print(output)
