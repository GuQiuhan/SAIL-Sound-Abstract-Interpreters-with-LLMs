import matplotlib.pyplot as plt
import numpy as np

# Define model names and operator labels
models = [
    "DeepSeek-V2-Lite",
    "Llama-3.3-70B-Instruct",
    ]

operators = [
    "Abs",
    "Affine",
    "Avgpool",
    "HardTanh",
    "HardSwish",
    "HardSigmoid",
    "Maxpool",
    "Relu",
]


# Runtime data (example values for  operators ×  models)
timings = [
    [10, 45, 38, 64, 34, 84, 61, 7],  # Model A
    [1,1,1,1,1,1,1,1],  # Model B
]


colors = ["#f6d1c1", "#f4a261"] # #e76f51 ~ https://colordrop.io/palette/34271

x = np.arange(len(operators))
bar_width = 0.35

plt.figure(figsize=(10, 6))

for i, model_times in enumerate(timings):
    plt.bar(x + i * bar_width, model_times, width=bar_width, label=models[i], color=colors[i])

# Labeling
plt.xlabel("Operator", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.title(f"Runtime per Operator for {len(models)} Models", fontsize=14)
plt.xticks(x + bar_width / len(models), operators, fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=11)

# Optional: Add gridlines
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("pics/gen_time.png", dpi=300)

