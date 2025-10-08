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

    CONSTRAINTFLOW = """
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepPoly transformers for DNN operators.
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
- func priority2(Neuron n, Float c) = -n[layer];
- func stop(Neuron n) = false;
- func stop_traverse(Neuron n, Float c) = false;
- func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, stop_traverse, replace_upper){e >= n}).map(simplify_upper);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
- func slope(Float x1, Float x2) = ((x1 * (x1 + 3))-(x2 * (x2 + 3))) / (6 * (x1-x2));
- func intercept(Float x1, Float x2) = x1 * ((x1 + 3) / 6) - (slope(x1, x2) * x1);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
- func f1(Float x) = x < 3 ? x * ((x + 3) / 6) : x;
- func f2(Float x) = x * ((x + 3) / 6);
- func f3(Neuron n) = max(f2(n[l]), f2(n[u]));
- func compute_l(Neuron n1, Neuron n2) = min([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func compute_u(Neuron n1, Neuron n2) = max([n1[l]*n2[l], n1[l]*n2[u], n1[u]*n2[l], n1[u]*n2[u]]);
- func avg(List<Float> xs) = sum(xs) / len(xs);
- func argmax(List<Neuron> ns, (Neuron, Neuron -> Bool) cmp) = [ n | n in ns, forall m in ns. cmp(n, m) ];
- func argmin(List<Neuron> ns, (Neuron, Neuron -> Bool) cmp) = [ n | n in ns, forall m in ns. cmp(n, m) ];

Don't add comments to DSL.
"""

    prmpt_relu = """
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
        Affine -> (backsubs_lower(prev.dot(curr[weight]) + curr[bias], curr), backsubs_upper(prev.dot(curr[weight]) + curr[bias], curr), prev.dot(curr[weight]) + curr[bias], prev.dot(curr[weight]) + curr[bias]);
    }
    """
    message = [
        {
            "role": "system",
            "content": "You are a formal methods expert working on neural network verification. Your task is to generate the DeepPoly transformers for DNN operators. Generate the transformer in Constraintflow DSL. {CONSTRAINTFLOW}",
        },
        {"role": "user", "content": "Generate the transformer for `relu` operator "},
        {"role": "assistant", "content": prmpt_relu},
        {"role": "user", "content": "Generate the transformer for `abs` operator "},
        {"role": "assistant", "content": prmpt_abs},
        {"role": "user", "content": "Generate the transformer for `relu6` operator "},
    ]

    prompt = f"""
    {CONSTRAINTFLOW}

    ### Example: ReLU operator
    Input: Generate the transformer for `relu` operator
    Output:
    {prmpt_relu}

    ### Example: Abs operator
    Input: Generate the transformer for `abs` operator
    Output:
    {prmpt_abs}

    ### Example: Affine operator
    Input: Generate the transformer for `affine` operator
    Output:
    {prmpt_affine}

    ### Now generate the transformer for `Sigmoid` operator
    Input: Generate the transformer for `Sigmoid` operator
    Output:
    """

    p = """
You are a DSL repair assistant. Fix the following DSL code based on the error.
[ERROR]:
Unknown keyword: let.
(If there exists any self-defined function, please directly use them)

[CODE]:
transformer deeppoly{
    Sigmoid -> (
        let l = 1 / (1 + exp(-prev[l]));
        let u = 1 / (1 + exp(-prev[u]));
        let L = (prev[u] - prev[l]) / (u - l) * prev[l] + prev[l];
        let U = (prev[u] - prev[l]) / (u - l) * prev[u] + prev[u];
        (l, u, L, U)
    );
}
"""
    output = client.chat(messages=message)
    # output = client.textgen(prompt=prompt)

    print(output)


'''
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


    message = [
    {"role": "system", "content": "You are a formal methods expert working on neural network verification. Your task is to generate the DeepPoly transformers for DNN operators. Generate the transformer in Constraintflow DSL. {CONSTRAINTFLOW}"},
    {"role": "user", "content": "Generate the transformer for `relu` operator "},
    {"role": "assistant", "content": prmpt_relu},
    {"role": "user", "content": "Generate the transformer for `abs` operator "},
    {"role": "assistant", "content": prmpt_abs},
    {"role": "user", "content": "Generate the transformer for `relu6` operator "},
]
    output = client.chat(messages = message)

  '''
