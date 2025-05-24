# LLM-based DSL Generation for Neuron Specification

> A framework for step-by-step DSL generation using Large Language Models (LLMs).

This project aims to automate the generation of **neuron-level DSL constraints** using prompting techniques such as Chain-of-Thought (CoT), verification-guided refinement, and multi-model comparison. The system supports flexible prompt design, modular model interfaces, and robust evaluation workflows.

---

## 📁 Project Structure

```
├── generation                # Core LLM generation logic
│   ├── prompt
│   │   ├── prompts           # (Few-shot) prompt examples and templates
│   │   └── doc_collector.py  # Collects operator documentation to support grounding
│   ├── controller.py         # Controls generation steps and retry logic
│   ├── models.py             # Unified model interface for Llama, Gemini, etc.
│   ├── request.py            # Prompt formatting and model communication
│   └── step_by_step.py       # Step-by-step constraint generation workflow
├── results/                  # Outputs of models' generation
│   ├── gemma-7b/
│   │   ├── success/
│   │   └── failure/
│   ├── llama3-1B/
│   │   ├── success/
│   │   └── failure/
│   └── ...
└── requirements.txt     # Python dependencies

```

## 🚀 Usage

### Configuration

#### Model Deployment
* Login in huggingface with your token, make sure have access to Llama3, Llama4, etc..
* Change the IP address of `MODEL_ENDPOINTS` in `generation/step_by_step.py` before deploying models.
### Documentation Collection
```bash
python generation/prompt/doc_collector.py
```
This tool scrapes and organizes PyTorch operator documentation for use in grounded prompting.

### Step-by-Step DSL Generation
```bash
python generation/step_by_step.py
```
This script guides the model to generate DSLs for neural operators using multi-stage reasoning and validation.



# TODO:

* read code of constraintflow and print out the counterexamples to prompt model
* package consranitflow
