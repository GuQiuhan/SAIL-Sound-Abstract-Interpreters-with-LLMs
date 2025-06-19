import matplotlib.pyplot as plt
import numpy as np

# Define model names and operator labels
models = ["DeepSeek-V2-Lite", "Llama-3.3-70B-Instruct", "GPT-4.1", "GPT-4o"]

"""
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
    [69,0,0,0,0,0,0,0],  # Model B
]
"""

operators = [
    "Abs",
    "Add",
    "Affine",
    "Avgpool",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "Max",
    "Maxpool",
    "Min",
    "Minpool",
    "Mult",
    "Relu",
    "Relu6",
]

results = [
    [  # DeepSeek
        ["Abs", 10, True],
        ["Add", 79, False],
        ["Affine", 45, True],
        ["Avgpool", 38, True],
        ["HardSigmoid", 84, True],
        ["HardSwish", 34, True],
        ["HardTanh", 64, True],
        ["Max", 70, False],
        ["Maxpool", 61, True],
        ["Min", 100, False],
        ["Minpool", 80, False],
        ["Mult", 67, False],
        ["Relu", 7, True],
        ["Relu6", 79, False],
    ],
    [  # Llama
        ["Abs", 69, True],
        ["Add", 814, False],
        ["Affine", 510, False],
        ["Avgpool", 547, False],
        ["HardSigmoid", 1107, False],
        ["HardSwish", 1625, False],
        ["HardTanh", 811, False],
        ["Max", 857, False],
        ["Maxpool", 954, False],
        ["Min", 1553, False],
        ["Minpool", 1140, False],
        ["Mult", 1414, False],
        ["Relu", 871, False],
        ["Relu6", 871, False],
    ],
    [  # GPT-4.1
        ["Abs", 130, False],
        ["Add", 74, False],
        ["Affine", 87, False],
        ["Avgpool", 87, False],
        ["HardSigmoid", 158, False],
        ["HardSwish", 162, False],
        ["HardTanh", 181, False],
        ["Max", 50, False],
        ["Maxpool", 143, False],
        ["Min", 128, False],
        ["Minpool", 119, False],
        ["Mult", 163, False],
        ["Relu", 7, True],
        ["Relu6", 211, False],
    ],
    [  # GPT-4o
        ["Abs", 45, True],
        ["Add", 83, False],
        ["Affine", 98, False],
        ["Avgpool", 103, False],
        ["HardSigmoid", 170, False],
        ["HardSwish", 200, False],
        ["HardTanh", 229, False],
        ["Max", 196, False],
        ["Maxpool", 150, False],
        ["Min", 113, False],
        ["Minpool", 113, False],
        ["Mult", 197, False],
        ["Relu", 10, True],
        ["Relu6", 251, False],
    ],
]


colors = [
    "#f6d1c1",
    "#f4a261",
    "#e76f51",
    "#2a9d8f",
]  # #e76f51 ~ https://colordrop.io/palette/34271

x = np.arange(len(operators)) * 1.8
bar_width = 0.35

plt.figure(figsize=(14, 6))

for model_idx, model_data in enumerate(results):
    times = [item[1] for item in model_data]
    successes = [item[2] for item in model_data]

    bars = plt.bar(
        x + model_idx * bar_width,
        times,
        width=bar_width,
        label=models[model_idx],
        color=colors[model_idx],
        zorder=1,
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        mark = "\u2713" if successes[i] else "\u2717"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            mark,
            ha="center",
            va="bottom",
            fontsize=12,
            zorder=3,
        )

# Labeling
plt.xlabel("Operator", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.title(
    f"Generation Time and Success per Operator for {len(models)} Models w.r.t. DeepPoly",
    fontsize=14,
)
plt.xticks(x + bar_width / len(models), operators, fontsize=7)
plt.yticks(fontsize=11)
plt.legend(fontsize=11)

# Optional: Add gridlines
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("pics/gen_time.png", dpi=300)
