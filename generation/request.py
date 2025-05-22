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

    DEEPPOLY_CONTEXT = """
Def shape as (Real l, Real u, PolyExp L, PolyExp U) {
    [curr[l] <= curr, curr[u] >= curr, curr[L] <= curr, curr[U] >= curr]
};

Func concretize_lower(Neuron n, Real c) = (c >= 0) ? (c * n[l]) : (c * n[u]);
Func concretize_upper(Neuron n, Real c) = (c >= 0) ? (c * n[u]) : (c * n[l]);

Func replace_lower(Neuron n, Real c) = (c >= 0) ? (c * n[L]) : (c * n[U]);
Func replace_upper(Neuron n, Real c) = (c >= 0) ? (c * n[U]) : (c * n[L]);

Func priority(Neuron n) = n[layer];

Func backsubs_lower(PolyExp e, Neuron n) = 
    (e.traverse(backward, priority, false, replace_lower){e <= n}).map(concretize_lower);

Func backsubs_upper(PolyExp e, Neuron n) = 
    (e.traverse(backward, priority, false, replace_upper){e >= n}).map(concretize_upper);

Transformer DeepPoly(curr, prev){
ReLU -> prev[l] > 0 ? (prev[l], prev[u], prev, prev) :
         (prev[u] < 0 ? (0, 0, 0, 0) :
         (0, prev[u], 0, ((prev[u] / (prev[u] - prev[l])) * prev) - ((prev[u] * prev[l]) / (prev[u] - prev[l]))));

Affine -> (
    backsubs_lower(prev.dot(curr[w]) + curr[b], curr),
    backsubs_upper(prev.dot(curr[w]) + curr[b], curr),
    prev.dot(curr[w]) + curr[b],
    prev.dot(curr[w]) + curr[b]
);
"""
    output = client.textgen(prompt = f"""
# DeepPoly DSL Transformer Generation

You are a formal methods expert writing a new transformer rule for a PyTorch operator in the DeepPoly DSL. Below is the abstract domain shape and two existing examples (ReLU and Affine). Now generate a new DSL transformer for the operator below.

{DEEPPOLY_CONTEXT}

API: torch.fft.fft
**********************
Documentation: 

torch.fft.fft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor¶
Computes the one dimensional discrete Fourier transform of input.

Note
The Fourier domain representation of any real signal satisfies the
Hermitian property: X[i] = conj(X[-i]). This function always returns both
the positive and negative frequency terms even though, for real inputs, the
negative frequencies are redundant. rfft() returns the
more compact one-sided representation where only the positive frequencies
are returned.


Note
Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
However it only supports powers of 2 signal length in every transformed dimension.


Parameters

input (Tensor) – the input tensor
n (int, optional) – Signal length. If given, the input will either be zero-padded
or trimmed to this length before computing the FFT.
dim (int, optional) – The dimension along which to take the one dimensional FFT.
norm (str, optional) – Normalization mode. For the forward transform
(fft()), these correspond to:

"forward" - normalize by 1/n
"backward" - no normalization
"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Calling the backward transform (ifft()) with the same
normalization mode will apply an overall normalization of 1/n between
the two transforms. This is required to make ifft()
the exact inverse.
Default is "backward" (no normalization).



Keyword Arguments
out (Tensor, optional) – the output tensor.


Example
>>> t = torch.arange(4)
>>> t
tensor([0, 1, 2, 3])
>>> torch.fft.fft(t)
tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])


>>> t = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
>>> torch.fft.fft(t)
tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
************************

Add your transformer below. Only generate the Transformer rule (no comments, no extra output):    
    
""")
    print(output)


'''
    output = client.textgen(prompt=f"""
# DeepPoly Compatibility Check

You are a formal methods expert working on neural network verification. Your task is to determine whether the following PyTorch operator is compatible with the 
DeepPoly abstract domain.

DeepPoly supports any operator for which:
- The output can be overapproximated using affine bounds (symbolic linear expressions).
- The operator is monotonic or piecewise-linear (e.g., ReLU, LeakyReLU, HardTanh).
- The operator can be decomposed into affine and element-wise operations (e.g., Add, Mul, Clamp).
- The operator acts in an element-wise or structured way (e.g., pooling, affine transforms).

DeepPoly does NOT support:
- Operators with non-elementwise behavior that cannot be soundly approximated with affine bounds.
- Non-monotonic or highly nonlinear operators like Softmax, Argmax, Dropout, Sampling, or control flow.

API: torch.fft.fft
**********************
Documentation: 

torch.fft.fft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor¶
Computes the one dimensional discrete Fourier transform of input.

Note
The Fourier domain representation of any real signal satisfies the
Hermitian property: X[i] = conj(X[-i]). This function always returns both
the positive and negative frequency terms even though, for real inputs, the
negative frequencies are redundant. rfft() returns the
more compact one-sided representation where only the positive frequencies
are returned.


Note
Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
However it only supports powers of 2 signal length in every transformed dimension.


Parameters

input (Tensor) – the input tensor
n (int, optional) – Signal length. If given, the input will either be zero-padded
or trimmed to this length before computing the FFT.
dim (int, optional) – The dimension along which to take the one dimensional FFT.
norm (str, optional) – Normalization mode. For the forward transform
(fft()), these correspond to:

"forward" - normalize by 1/n
"backward" - no normalization
"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Calling the backward transform (ifft()) with the same
normalization mode will apply an overall normalization of 1/n between
the two transforms. This is required to make ifft()
the exact inverse.
Default is "backward" (no normalization).



Keyword Arguments
out (Tensor, optional) – the output tensor.


Example
>>> t = torch.arange(4)
>>> t
tensor([0, 1, 2, 3])
>>> torch.fft.fft(t)
tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])


>>> t = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
>>> torch.fft.fft(t)
tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
************************

Generation Requirement:
Just output a number!!
- `1` if the operator is supported.
- `0` if the operator is not supported.
    """)
'''