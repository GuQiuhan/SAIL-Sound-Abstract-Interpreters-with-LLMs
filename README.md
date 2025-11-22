# ⛵ Cost-Driven Synthesis of Sound Abstract Interpreters

[![arXiv](https://img.shields.io/badge/arXiv-2511.13663-b31b1b.svg)](https://arxiv.org/abs/2511.13663)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/github/license/GuQiuhan/SAIL-Sound-Abstract-Interpreters-with-LLMs)
![Stars](https://img.shields.io/github/stars/GuQiuhan/SAIL-Sound-Abstract-Interpreters-with-LLMs?style=social)

![workflow](experiments/run_tables/general_workflow.png)

> The overview of SAIL.

How to construct **globally sound** abstract interpreters to safely approximate program behaviors remains a bottleneck in abstract interpretation. We show the potential of using state-of-the-art LLMs to automate this tedious process. Focusing on the neural network verification area, we synthesize non-trivial sound abstract transformers across diverse abstract domains using LLMs to search within **infinite space** from scratch. We formalize the synthesis task as a constrained optimization problem, for which we design a novel mathematically grounded cost function that measures the degree of unsoundness of each generated candidate transformer, while enforcing hard syntactic and semantic validity constraints.

---

## 📁 Project Structure

```
├── generation                # Core LLM generation logic
│   ├── validator
│   │   ├── miniDSL           # ANTLR grammar files and parser modules for the DSL
│   │   │   └── ...
│   │   ├── repair            # LLM-driven repair logic for incorrect DSL generations
│   │   ├── syntax_check      # Rule-based syntax checker and fixer for malformed DSL code
│   │   └── semantics_check   # Type-based semantic checker for DSL AST
│   │   └── soundness_check   # Check global soundness based on SMT
│   ├── evaluator
│   │   ├── eval              # Compute the cost function for each candidate
│   ├── run_all.py            # One-click launcher: starts model server, runs generation, shuts down
│   ├── models.py             # Unified model interface for Llama, Gpt, DeepSeek, etc.
│   ├── request.py            # Prompt formatting and model communication
│   ├── gen.py                # Constraint generation workflow, allow multiple models and multiple certifiers
│   └── utils.py              # Shared utilities and constants (e.g., model-port mapping, helper functions)
│   └──logs/                  # Outputs of models' generation, including generation results, generation log, statistic analysis
│   │   ├── date1/
│   │   │   ├──model1/
│   │   │   │   ├── certifier1/
│   │   │   │   │   └── failure/
│   │   │   │   │   └── success/
│   │   │   │   │   └── statistics/
│   │   │   │   │   │   └── statistics.json
│   │   │   │   │   │   └── xxx.png
│   │   │   │   │   │   └── op_monitor.json
│   │   │   │   │   └── generation.log
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
└── constraintflow/      # constraintflow engine
└── experiments/         # evaluations in the paper
└── requirements.txt     # Python dependencies
└── setup.py             # Pack the project

```

## 🚀 Usage

### 🔨 Project Configuration

#### Constraintflow Configuration
```bash
pip install -r requirements.txt
pip install -e .
```
#### Model Configuration
* Login AWS with your token, make sure have access to Llama3, Llama4, etc..
* Login OpenAI with your key, so that you can access GPT-5, GPT-4o.
* Change the IP address of `MODEL_PORT_PAIRS` and `PORT_MAP` in `generation/utils.py` before deploying models.


### 📦 Run the Pipeline

#### 1. Model Deployment
Before running `generation/gen.py`, you must start the model server. You can launch one or more supported LLMs (e.g., GPT, LLaMA, GPT) via:

```bash
python generation/models.py --model MODEL1[,MODEL2,...]
# or equivalently:
python generation/models.py -m MODEL1[,MODEL2,...]

```

#### Supported Model Options:
The `--model`/`-m` flag accepts any of the following model identifiers:
- llama
- llama3.3
- llama4
- deepseek
- gpt-4o
- gpt-4.1
- o4-mini
- gpt-5
- gpt-oss
- jamba
- titan
- nova
- claude
- mistral

This will start a local Flask server on the specified port, allowing `gen.py` to interact with the LLM.

> ⚠️ In the paper we just use gpt-5, gpt-4o, llama4-maverick, claude-opus-4 these four models. You can feel free to try any other models here, as long as the `MODEL_PORT_PAIRS` and `PORT_MAP` in `generation/utils.py` are correctly configured with your machine's IP and desired ports.

#### 2. Transformer Synthesis

```bash
python generation/gen.py -m [MODELS] -c [CERTIFIERS]
```
This script guides the model to synthesis transformers for neural operators in an iterative way, combining syntactic and semantic validation and cost_based quantative feedback.

#### Parameters

| Argument             | Type                    | Default     | Description                                                                                                   |
|----------------------|-------------------------|-------------|---------------------------------------------------------------------------------------------------------------|
| `--model`, `-m`      | `str` (multiple allowed)| `deepseek`  | One or more model keywords to launch. Options: `llama3.3`, `llama4`, `deepseek`, `gpt-4o`, `gpt-4.1`, `o4-mini`, `gpt-5`, `gpt-oss`, `jamba`, `titan`, `nova`, `claude`, `mistral`. Make sure you already launched the model in the server before in the step1. |
| `--certifier`, `-c`  | `str`                   | `deeppoly`  | One or more domains to synthesis. Options: `deeppoly`, `ibp` ,`deepz`                                                 |

#### Example Usage
```bash
python generation/gen.py --model gpt-5 --certifier deeppoly

# Run multiple models at once
python generation/run_all.py -m deepseek llama-3.3 -c deepz

# Run multiple certifiers at once
python generation/gen.py -m gpt-5 -c deeppoly, deepz, ibp

# Run multiple models and multiple certifiers at once.
# In this case there will be 3x3=9 combinations.
python generation/run_all.py -m gpt-5, claude, llama4 -c deeppoly, deepz, ibp
```

#### 3. You can also test each module separately:

#### - Validation and Repair Module
This project integrates a three-stage validation and repair framework.

* Syntax Checker (`generation/validator/syntax_check.py`)
Performs static structural validation of DSL code and automatically repairs common syntax issues.
```bash
python generation/validator/syntac_check.py
```

* Semantic Checker (`generation/validator/semantics_check.py`)
Performs type-based semantic analysis over parsed DSL AST.
```bash
python generation/validator/semantics_check.py
```
* LLM-Guided Repair (`generation/validator/repair.py`)
When the previous two parts detect the erros and fail to fix, the generation and errors are injected into the next LLM prompt.
```bash
python generation/validator/repair.py
```
In this repair submodule, you can feel free to change the model used to repair in `generation/validator/repair.py`.

#### - Soundness Verification Module
Given a valid transformer DSL, you can verify its soundness and get counterexampels and the cost-based score if it's unsound.
```bash
python -m generation.evaluator.eval
```

### 📈 Evaluation
#### 1. Train neural networks for testing the precision
```bash
cd experiments
cd model_training
python models_elu.py
python models_gelu.py
python models_hardsigmoid.py
python models_hardswish.py
python models_hardtanh.py
python models_relu6.py
python models_selu.py
```
These will save both fully connected (FCN) and convolutional (Conv) networks trained on MNIST and CIFAR10 datasets to `experiments/model_training/checkpoints/`.

Alternatively, you can download the networks [here](https://drive.google.com/drive/folders/1WhFoLtKhehkyeEChA1DQx7v16DBBrJjB?usp=sharing) directly, put them in `experiments/run_precision/nets/`.

#### 2. Test precision
When testing precision with the networks we trained, for MNIST, we apply the perturbation to the entire image, whereas for CIFAR10 we restrict the perturbation to a single pixel to reflect more localized robustness settings.

Make sure you put networks in `experiments/run_precision/nets/`.

Choose the constraintflow(.cf) file you wanna test under `experiments/run_precision/`, change the corresbonding parameter in the `main` function in `experiments/run_precision/experiments_precision.py`.

To change the perturbation and dataset, go to `experiments/run_precision/experiments_precision.py` and change the parameter in the `main` function.

If you are testing the precision for `Relu6`, `HardTanh`, `Gelu`, go to `constraintflow/core/lib/parse.py`, change the flag in line 108,109,110 respectively first.

Go to `constraintflow/core/lib/spec.py`, uncomment line 285,286 or line 289,290 based on the dataset you are testing.

Upon finishing the setting, run
```bash
experiments/run_precision/experiments_precision.py
```

Results will be saved in `experiments/run_precision/precision_results`.

#### 3. Run ablation study
```bash
cd experiments
cd run_ablation
python gen_wo_cex.py
```
