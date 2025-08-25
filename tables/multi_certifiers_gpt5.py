import os

import matplotlib.pyplot as plt
import numpy as np

raw_data = [
    ["Abs", 1, 1, 1, 14.23, True],
    ["Add", 1, 0, 0, 5.94, True],
    ["Affine", 10, 52, 0, 293.83, False],
    ["Avgpool", 10, 60, 0, 582.94, True],
    ["HardSigmoid", 10, 7, 22, 297.40, True],
    ["HardSwish", 10, 28, 17, 451.92, False],
    ["HardTanh", 4, 13, 5, 209.80, True],
    ["Max", 2, 2, 2, 66, True],
    ["Maxpool", 1, 17, 0, 74.68, True],
    ["Min", 3, 12, 4, 122.56, True],
    ["Minpool", 9, 58, 0, 312.64, True],
    ["Mult", 10, 86, 0, 1266.64, False],
    ["Relu", 1, 0, 0, 8.56, True],
    ["Relu6", 10, 5, 21, 407.79, True],
    [0],
]


def rounds_subplot(ax, data, model_name):
    total_time = data[-1][0] if data[-1] else None
    raw_data = data[:-1]
    operators = [item[0] for item in raw_data]
    gen_rounds = [item[1] for item in raw_data]
    repair_rounds = [item[2] for item in raw_data]
    ce_counts = [item[3] for item in raw_data]
    times = [item[4] for item in raw_data]
    success_flags = [item[5] for item in raw_data]

    x = np.arange(len(operators))
    bar_width = 0.2

    # Bar charts
    bars1 = ax.bar(
        x - bar_width,
        gen_rounds,
        width=bar_width,
        label="LLM Gen Rounds",
        color="#f7b7a3",
    )
    bars2 = ax.bar(
        x,
        repair_rounds,
        width=bar_width,
        label="LLM Repair Rounds",
        color="#f6d5e3",
    )
    bars3 = ax.bar(
        x + bar_width,
        ce_counts,
        width=bar_width,
        label="Counterexamples",
        color="#7e9ce0",
    )

    ax.set_title(f"{model_name}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(operators, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rounds / CEs", fontsize=8)

    # Secondary y-axis for time
    ax2 = ax.twinx()
    ax2.plot(x, times, color="#4b6f8d", marker="o", linewidth=1, label="Time (s)")
    ax2.set_ylabel("Time (s)", fontsize=8)

    # Add bar labels
    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(height),
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Add success/fail marks
    for i in range(len(operators)):
        max_height = max(gen_rounds[i], repair_rounds[i], ce_counts[i])
        symbol = "✓" if success_flags[i] else "✗"
        ax.text(x[i], max_height + 3, symbol, ha="center", va="bottom", fontsize=8)

    # Adjust Y limits to avoid overlaps
    max_height = max(
        max(gen_rounds, default=0),
        max(repair_rounds, default=0),
        max(ce_counts, default=0),
    )
    ax.set_ylim(top=max_height + 15)

    # Total runtime annotation
    if total_time is not None:
        ax.text(
            0.5,
            -0.25,
            f"Total runtime: {total_time:.2f}s",
            ha="center",
            va="top",
            fontsize=7,
            transform=ax.transAxes,
        )


def draw_three(statistics, output_path):
    num_models = len(statistics)

    rows, cols = num_models, 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6 * rows))

    if num_models == 1:
        axes = [axes]

    for idx, (model_name, data) in enumerate(statistics.items()):
        rounds_subplot(axes[idx], data, model_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)

    plt.suptitle(
        "LLM Rounds, Counterexamples, and Time per Operator (Model: GPT-5)", fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Suppose you have 9 datasets in a dict
    statistics = {
        "DeepPoly": [
            ["Abs", 1, 0, 0, 128.96567606925964, True],
            ["Affine", 2, 0, 0, 830.1170408725739, True],
            ["Avgpool", 10, 6, 0, 1559.7766621112823, True],
            ["HardSigmoid", 1, 3, 1, 903.027379989624, True],
            ["HardSwish", 10, 0, 1, 26696.940661907196, True],
            ["HardTanh", 1, 0, 0, 369.2898533344269, True],
            ["Maxpool", 1, 1, 0, 291.0352168083191, True],
            ["Minpool", 1, 3, 0, 540.1951503753662, True],
            ["Neuron_add", 1, 0, 0, 60.26173758506775, True],
            ["Neuron_max", 1, 0, 0, 180.59920454025269, True],
            ["Neuron_min", 1, 0, 0, 188.7460196018219, True],
            ["Neuron_mult", 1, 3, 0, 531.7936959266663, True],
            ["Relu", 1, 0, 0, 119.0497395992279, True],
            ["Relu6", 1, 0, 0, 428.9358389377594, True],
            [32828.75975847244],
        ],
        "IBP": [
            ["Abs", 1, 0, 0, 78.77562499046326, True],
            ["Affine", 10, 42, 0, 6410.81347990036, True],
            ["Avgpool", 10, 54, 0, 4947.610588550568, True],
            ["HardSigmoid", 2, 6, 2, 813.1757967472076, True],
            ["HardSwish", 3, 0, 1, 6315.879325866699, True],
            ["HardTanh", 1, 0, 0, 77.98450779914856, True],
            ["Maxpool", 1, 1, 0, 141.08218216896057, True],
            ["Minpool", 4, 8, 0, 2536.98224234581, True],
            ["Neuron_add", 1, 0, 0, 56.18330216407776, True],
            ["Neuron_max", 8, 62, 0, 3284.918600320816, True],
            ["Neuron_min", 1, 0, 0, 46.693279504776, True],
            ["Neuron_mult", 10, 87, 0, 6118.334964990616, True],
            ["Relu", 1, 0, 0, 40.62461471557617, True],
            ["Relu6", 1, 0, 0, 73.03750205039978, True],
            [30942.104109287262],
        ],
        "DeepZ": [
            ["Abs", 1, 1, 0, 150.09745383262634, True],
            ["Affine", 10, 35, 1, 7261.917321920395, False],
            ["Avgpool", 7, 26, 1, 2848.151502609253, True],
            ["HardSigmoid", 1, 4, 0, 1818.8842465877533, True],
            ["HardSwish", 10, 1, 2, 17718.102370023727, False],
            ["HardTanh", 1, 0, 0, 149.28806614875793, True],
            ["Maxpool", 1, 0, 0, 149.8353409767151, True],
            ["Minpool", 4, 6, 0, 1196.5830013751984, True],
            ["Neuron_add", 1, 0, 0, 42.47822308540344, True],
            ["Neuron_max", 1, 0, 0, 119.66511654853821, True],
            ["Neuron_min", 1, 0, 0, 79.33962059020996, True],
            ["Neuron_mult", 1, 1, 0, 176.42982244491577, True],
            ["Relu", 1, 0, 0, 239.47893571853638, True],
            ["Relu6", 1, 0, 0, 126.3460922241211, True],
            [32076.604518175125],
        ],
    }
    draw_three(statistics, "pics/multi_certifiers_gpt5.png")
