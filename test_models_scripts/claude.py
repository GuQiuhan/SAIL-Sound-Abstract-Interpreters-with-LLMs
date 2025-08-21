import json
import os

import boto3

CONSTRAINTFLOW = """
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
- func priority2(Neuron n) = -n[layer];
- func backsubs_lower(PolyExp e, Neuron n) = (e.traverse(backward, priority2, true, replace_lower){e <= n}).map(simplify_lower);
- func backsubs_upper(PolyExp e, Neuron n) = (e.traverse(backward, priority2, true, replace_upper){e >= n}).map(simplify_upper);
- func f(Neuron n1, Neuron n2) = n1[l] >= n2[u];
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
    "messages": convert_system_to_user_with_ack(message),
}


response = bedrock.invoke_model(
    modelId="us.anthropic.claude-opus-4-20250514-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body),
)


response_body = json.loads(response["body"].read())
output_text = response_body["content"][0]["text"]

print(output_text)
