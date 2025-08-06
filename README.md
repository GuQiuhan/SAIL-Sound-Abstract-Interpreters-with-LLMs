# LLM-based DSL Generation for Abstract Transformer

![workflow](https://github.com/GuQiuhan/ConstraintFlow_patch1/blob/main/tables/pics/workflow.png)

> A framework for DSL generation using Large Language Models (LLMs).

This project aims to automate the generation of **neuron-level DSL constraints** using prompting techniques such as Chain-of-Thought (CoT), verification-guided refinement, and multi-model comparison. The system supports flexible prompt design, modular model interfaces, and robust evaluation workflows.

---

## 📁 Project Structure

```
├── generation                # Core LLM generation logic
│   ├── prompt
│   │   ├── prompts           # (Few-shot) prompt examples and templates
│   │   └── doc_collector.py  # Collects operator documentation to support grounding
│   ├── validator
│   │   ├── miniDSL           # ANTLR grammar files and parser modules for the DSL
│   │   │   └── ...
│   │   ├── repair            # LLM-driven repair logic for incorrect DSL generations
│   │   ├── syntax_check      # Rule-based syntax checker and fixer for malformed DSL code
│   │   └── semantics_check   # Type-based semantic checker for DSL AST
│   ├── evaluator
│   │   ├── eval              # Compute the cost function for each candidate
│   ├── run_all.py            # One-click launcher: starts model server, runs generation, shuts down
│   ├── models.py             # Unified model interface for Llama, Gpt, DeepSeek, etc.
│   ├── request.py            # Prompt formatting and model communication
│   ├── gen.py                # Constraint generation workflow, allow multiple models and multiple certifiers
│   └── reasoning_gen.py      # Constraint generation workflow augmented with reasoning steps
│   └── utils.py              # Shared utilities and constants (e.g., model-port mapping, helper functions)
├── results/                  # Outputs of models' generation, including generation results, generation log, statistic analysis
│   ├── date1/
│   │   ├──deepseek/
│   │   │   ├── certifier/
│   │   │   │   └── failure/
│   │   │   │   └── success/
│   │   │   │   └── statistics/
│   │   │   │   │   └── statistics.json
│   │   │   │   │   └── statistics.png
│   │   │   │   └── generation.log
│   ├── date2/
│   │   ├──llama/
│   │   │   ├── certifier/
│   │   │   │   └── failure/
│   │   │   │   └── success/
│   │   │   │   └── statistics/
│   │   │   │   └── generation.log
│   └── ...
└── requirements.txt     # Python dependencies
└── setup.py             # Pack the project

```

## 🚀 Usage

### 🔨 Configuration

#### Constraintflow Configuration
```bash
pip install -e .
```
#### Model Configuration
* Login in huggingface with your token, make sure have access to Llama3, Llama4, etc..
* Change the IP address of `MODEL_ENDPOINTS` in `generation/utils.py` before deploying models.


### 📦 All-in-One Pipeline (Model Deployment + DSL Generation)
```bash
python generation/run_all.py --model MODEL_NAME --certifier CERTIFIER_NAME
```
#### Parameters

| Argument             | Type                    | Default     | Description                                                                                                   |
|----------------------|-------------------------|-------------|---------------------------------------------------------------------------------------------------------------|
| `--model`, `-m`      | `str` (multiple allowed)| `deepseek`  | One or more model keywords to launch. Options: `llama-3.3`, `llama-4`, `deepseek`, `gpt-4o`, `gpt-4.1`, `o4-mini` |
| `--certifier`, `-c`  | `str`                   | `deeppoly`  | Type of certifier to use. Options: `deeppoly`, `ibp`, `deepz`                                                 |

#### Example Usage
```bash
# Run with default DeepSeek model and DeepPoly certifier
python generation/run_all.py

# Specify model and certifier
python generation/run_all.py --model llama-4 --certifier ibp

# Run multiple models at once
python generation/run_all.py -m deepseek llama-3.3 -c deepz
```



### 📖 Documentation Collection
```bash
python generation/prompt/doc_collector.py
```
This tool scrapes and organizes PyTorch operator documentation for use in grounded prompting.

### 🧠 Model Deployment
Before running `gen.py`, you must start the model server. You can launch any supported LLMs (e.g., DeepSeek, LLaMA3/4, GPT-4o) via:

```bash
python generation/models.py --model MODEL_NAME
```

**Supported model options**:
- `deepseek`
- `llama-3.3`
- `llama-4`
- `gpt-4o`
- `gpt-4.1`
- `o4-mini`

> ⚠️ Make sure the `MODEL_ENDPOINTS` in `utils.py` is correctly configured with your machine's IP and desired ports.

This will start a local Flask server on the specified port, allowing `gen.py` to interact with the LLM.

### 🖨️ DSL Generation
```bash
python generation/gen.py
```
This script guides the model to generate DSLs for neural operators using multi-stage reasoning and validation.

#### 🛠️ DSL Validation and Repair Pipeline
This project integrates a three-stage validation and repair framework for dDSL generation.

* Syntax Checker (`generation/validator/syntax_check.py`)
Performs static structural validation of DSL code and automatically repairs common syntax issues.
```bash
python -m generation.validator.syntac_check
```

* Semantic Checker (`generation/validator/semantics_check.py`)
Performs type-based semantic analysis over parsed DSL AST.
```bash
python -m generation.validator.semantics_check
```
* LLM-Guided Repair (`generation/validator/repair.py`)
When the previous two parts detect the erros and fail to fix, the generation and errors are injected into the next LLM prompt.
```bash
python -m generation.validator.repair
```

# TODO:

* [x] read code of constraintflow and print out the counterexamples to prompt model
* [x] package constraitflow
* [ ] reasoning for `join` and `meet` operators
* [x] llm repair
* [x] the controller/orchestrator
* [ ] improve dsl validation module
* [ ] formalize the generation, verification, repair phases into algorithms
* [ ] formal proofs of the soundness, completeness, (efficiency) of proposed algorithms
* [ ] evaluations: compare with other transformer generation/synthesis baselines
* [ ] Analysis: - Can it generate more complicated transformers?
* [ ] Fix constraintflow, including negative floats/ProveSound
