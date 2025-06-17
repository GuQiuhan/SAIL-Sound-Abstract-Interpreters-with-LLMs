# LLM-based DSL Generation for Neuron Specification

> A framework for DSL generation using Large Language Models (LLMs).

This project aims to automate the generation of **neuron-level DSL constraints** using prompting techniques such as Chain-of-Thought (CoT), verification-guided refinement, and multi-model comparison. The system supports flexible prompt design, modular model interfaces, and robust evaluation workflows.

---

## 📁 Project Structure

```
├── generation                # Core LLM generation logic
│   ├── prompt
│   │   ├── prompts           # (Few-shot) prompt examples and templates
│   │   └── doc_collector.py  # Collects operator documentation to support grounding
│   ├── run_all.py            # One-click launcher: starts model server, runs generation, shuts down
│   ├── models.py             # Unified model interface for Llama, Gpt, DeepSeek, etc.
│   ├── request.py            # Prompt formatting and model communication
│   └── gen.py                # Constraint generation workflow
│   └── utils.py              # Shared utilities and constants (e.g., model-port mapping, helper functions)
├── results/                  # Outputs of models' generation
│   ├── date1/
│   │   ├──deepseek/
│   │   │   ├── success/
│   │   │   └── failure/
│   ├── date2/
│   │   ├──llama/
│   │   │   ├── success/
│   │   │   └── failure/
│   └── ...
└── requirements.txt     # Python dependencies

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



# TODO:

* [x] read code of constraintflow and print out the counterexamples to prompt model
* [x] package constraitflow 
