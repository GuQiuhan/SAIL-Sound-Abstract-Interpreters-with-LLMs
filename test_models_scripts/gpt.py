import os

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""

client = OpenAI()

CONSTRAINTFLOW = """
DeepPoly certifier uses four kinds of bounds to approximate the operator: (Float l, Float u, PolyExp L, PolyExp U).
They must follow the constraints that: curr[l] <= curr <= curr[u] & curr[L] <= curr <= curr[U]. `curr` here means the current neuron, `prev` means the inputs to the operator.
So every transformer in each case of the case analysis must return four values.
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

response = client.responses.create(
    model="o4-mini",
    input=message,
)

print(response.output_text)
