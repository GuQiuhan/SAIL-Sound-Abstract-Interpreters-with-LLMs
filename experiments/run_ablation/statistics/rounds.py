import os

import matplotlib.pyplot as plt
import numpy as np

# op, gen_rounds, repair_reounds, ce_number, time, success/fail
d1 = [
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
]


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
    ["Relu6", 10, 5, 21, 407.79, False],
    [0],
]


def rounds(data, path, model):
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

    fig, ax1 = plt.subplots(figsize=(14, 6))
    # fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    bars1 = ax1.bar(
        x - bar_width,
        gen_rounds,
        width=bar_width,
        label="LLM Gen Rounds",
        color="#f7b7a3",
    )
    bars2 = ax1.bar(
        x, repair_rounds, width=bar_width, label="LLM Repair Rounds", color="#f6d5e3"
    )
    bars3 = ax1.bar(
        x + bar_width,
        ce_counts,
        width=bar_width,
        label="Counterexamples",
        color="#7e9ce0",
    )

    ax1.set_ylabel("Number of Rounds / CEs")
    ax1.set_xlabel("Operators")
    ax1.set_xticks(x)
    ax1.set_xticklabels(operators, rotation=45, ha="right")

    ax2 = ax1.twinx()
    line = ax2.plot(
        x, times, color="#4b6f8d", marker="o", label="Time (s)", linewidth=2
    )
    ax2.set_ylabel("Time (s)")

    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_labels[0] + lines_labels2[0],
        lines_labels[1] + lines_labels2[1],
        loc="upper left",
    )

    max_height = max(
        max(gen_rounds, default=0),
        max(repair_rounds, default=0),
        max(ce_counts, default=0),
    )
    ax1.set_ylim(top=max_height + 15)

    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    for i in range(len(operators)):
        max_height = max(gen_rounds[i], repair_rounds[i], ce_counts[i])
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax1.text(x[i], max_height + 5, symbol, ha="center", va="bottom", fontsize=12)

    plt.title("LLM Rounds, Counterexamples, and Time per Operator")

    # plt.tight_layout()
    if total_time is not None:
        fig.text(
            0.5,
            0.02,
            f"Total runtime for model {model}: {total_time:.2f} seconds",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig(path, dpi=300)


def draw_all(statistic, dir):
    for model_name, data in statistic.items():
        path = os.path.join(dir, f"{model_name}.png")
        rounds(data, path, model_name)


if __name__ == "__main__":
    d = [
        ["Abs", 1, 0, 0, 20.062506914138794, True],
        ["Affine", 1, 0, 0, 28.318931579589844, True],
        [48.386308908462524],
    ]
    rounds(d1, "tmp.png", "gpt-5")
