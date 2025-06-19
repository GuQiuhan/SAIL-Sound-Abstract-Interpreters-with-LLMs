import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from vllm import LLM, SamplingParams

llm = LLM(
    model="/share/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=8192,
)

print("✅ Model loaded. Ready for generation.")

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
    top_p=1.0,
)

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

prompt = f"""
You are a formal methods expert working on neural network verification.
Your task is to generate the DeepPoly transformers for DNN operators.
Generate the transformer in Constraintflow DSL.

{CONSTRAINTFLOW}

### Example: ReLU operator
Input: Generate the transformer for `relu` operator
Output:
{prmpt_relu}

### Example: Abs operator
Input: Generate the transformer for `abs` operator
Output:
{prmpt_abs}

### Now generate the transformer for `relu6` operator
Input: Generate the transformer for `relu6` operator
Output:
"""

outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)

for output in outputs:
    print(output.outputs[0].text)


# works well
"""generation:
```constraintflow
def Shape as (Float l, Float u, PolyExp L, PolyExp U){[(curr[l]<=curr),(curr[u]>=curr),(curr[L]<=curr),(curr[U]>=curr)]};

transformer deeppoly{
    Relu6 -> ((prev[l]) >= 0) ? ((prev[l]), (prev[u]), (prev), (prev)) : (((prev[u]) <= 6) ? (((prev[u]) <= 0) ? (0, 0, 0, 0) : ((prev[u]) <= 6) ? ((prev[u]), (prev[u]), (prev), (prev)) : (0, 6, 0, 6)) : (((prev[u]) > 6) ? (0, 6, 0, 6) : (0, 0, 0, 0)) );
}
```

This transformer for `relu6` operator will return four values based on the input bounds, following the DeepPoly constraints.
"""
