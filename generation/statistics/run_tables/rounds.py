import os

import matplotlib.pyplot as plt
import numpy as np

# op, gen_rounds, repair_reounds, ce_number, time, success/fail
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

r2 = [
    ["Abs", 1, 1, 0, 20.15779972076416, True],
    ["Affine", 1, 3, 0, 51.69260311126709, True],
    ["Avgpool", 8, 6, 0, 433.80144691467285, True],
    ["HardSigmoid", 6, 0, 5, 225.05981183052063, True],
    ["HardSwish", 3, 0, 3, 120.83455085754395, True],
    ["HardTanh", 4, 1, 1, 266.952449798584, True],
    ["Maxpool", 1, 7, 0, 74.68, True],
    ["Minpool", 9, 10, 0, 312.64, True],
    ["Neuron_add", 1, 0, 0, 10.534726619720459, True],
    ["Neuron_max", 2, 3, 1, 51.066542863845825, True],
    ["Neuron_min", 2, 3, 1, 69.6841311454773, True],
    ["Neuron_mult", 10, 10, 0, 273.89032077789307, True],
    ["Relu", 1, 0, 0, 13.895500183105469, True],
    ["Relu6", 10, 5, 6, 407.79, True],
    [2951.7467756271362],
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
            0.02,  # y 设置为 0.02 而不是负值（保持在图像内部靠下）
            f"Total runtime for model: {total_time:.2f} seconds",
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
    rounds(raw_data, "pics/rounds_wo.png", "gpt-4o")
