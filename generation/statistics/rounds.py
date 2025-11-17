import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    ["Relu6", 10, 5, 21, 407.79, False],
    [0],
]


gpt5_deeppoly = [
    ["Abs", 1, 3, 0, 417.3745975494385, True],
    ["Affine", 10, 23, 0, 3023.0190472602844, False],
    ["Avgpool", 10, 19, 0, 2066.090816259384, False],
    ["HardSigmoid", 1, 0, 0, 336.812641620636, True],
    ["HardSwish", 4, 7, 0, 7274.458252191544, True],
    ["HardTanh", 1, 0, 1, 281.54825830459595, True],
    ["Maxpool", 1, 1, 0, 233.76632356643677, True],
    ["Minpool", 4, 0, 0, 1787.2058565616608, True],
    ["Neuron_add", 2, 0, 0, 69.1942241191864, True],
    ["Neuron_max", 1, 0, 0, 107.19696164131165, True],
    ["Neuron_min", 1, 0, 0, 99.59152913093567, True],
    ["Neuron_mult", 1, 4, 0, 285.4137122631073, True],
    ["Relu", 1, 0, 0, 61.4987678527832, True],
    ["Relu6", 1, 0, 0, 176.70161747932434, True],
    [16219.883420944214],
]


gpt5_deepz = [
    ["Abs", 1, 0, 0, 97.10799384117126, True],
    ["Affine", 10, 15, 0, 6257.354681491852, False],
    ["Avgpool", 10, 53, 2, 4606.029059648514, False],
    ["HardSigmoid", 1, 1, 0, 563.9685323238373, True],
    ["HardSwish", 10, 3, 1, 26309.281621217728, False],
    ["HardTanh", 1, 0, 0, 161.3286428451538, True],
    ["Maxpool", 1, 0, 0, 170.80931305885315, True],
    ["Minpool", 10, 4, 0, 3565.4361758232117, False],
    ["Neuron_add", 1, 0, 0, 75.31101584434509, True],
    ["Neuron_max", 1, 0, 0, 338.4349014759064, True],
    ["Neuron_min", 1, 0, 0, 131.95950603485107, True],
    ["Neuron_mult", 1, 2, 0, 268.67710876464844, True],
    ["Relu", 2, 2, 1, 543.8428502082825, True],
    ["Relu6", 1, 0, 0, 177.38560605049133, True],
    [43266.95149946213],
]


claude_deeppoly = [
    ["Abs", 7, 0, 3, 270.04334020614624, True],
    ["Affine", 15, 0, 0, 538.8747441768646, False],
    ["Avgpool", 15, 72, 0, 4079.348925590515, False],
    ["HardSigmoid", 40, 3, 9, 843.0143072605133, True],
    ["HardSwish", 80, 103, 14, 6806.651964902878, False],
    ["HardTanh", 45, 3, 7, 907.9705202579498, False],
    ["Maxpool", 15, 64, 0, 3471.6564424037933, False],
    ["Minpool", 15, 59, 0, 3313.0274620056152, False],
    ["Neuron_add", 1, 0, 0, 28.052807331085205, True],
    ["Neuron_max", 1, 0, 0, 58.71096968650818, True],
    ["Neuron_min", 2, 0, 1, 117.04786205291748, True],
    ["Neuron_mult", 15, 115, 0, 12164.76506781578, False],
    ["Relu", 1, 0, 0, 28.694417476654053, True],
    ["Relu6", 15, 0, 1, 830.9136326313019, False],
    [33458.78201794624],
]


def rounds_gen_only(data, path, model):
    total_time = data[-1][0] if data[-1] else None

    raw_data = data[:-1]
    operators = [item[0] for item in raw_data]
    gen_rounds = [item[1] for item in raw_data]
    success_flags = [item[5] for item in raw_data]

    x = np.arange(len(operators)) * 1.4
    bar_width = 0.3

    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.subplots_adjust(bottom=0.25)

    ax1.set_facecolor("#f0f0f0")
    ax1.grid(True, axis="y", color="white", linewidth=1.2)
    ax1.set_axisbelow(True)

    bars = ax1.bar(
        x,
        gen_rounds,
        width=bar_width,
        label="LLM Gen Rounds",
        color="#f7b7a3",
    )

    ax1.set_ylabel("Numbers", fontsize=10, labelpad=10, rotation=0, loc="top")
    ax1.yaxis.set_label_coords(0.037, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels(operators, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelsize=10)

    # extra_handles = [
    #     Line2D([], [], color="black", marker=r"$✓$", markersize=12, linestyle="None", label="Sound"),
    #     Line2D([], [], color="black", marker=r"$✗$", markersize=12, linestyle="None", label="Unsound")
    # ]
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(
    #     handles + extra_handles,
    #     labels + ["Sound Generation", "Unsound Generation"],
    #     loc="upper right",
    #     fontsize=11
    # )

    max_height = max(gen_rounds, default=0)
    ax1.set_yticks(np.arange(0, max_height + 15, 35))

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for i in range(len(operators)):
        max_height = gen_rounds[i]
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax1.text(x[i], max_height + 3, symbol, ha="center", va="bottom", fontsize=12)

    for spine in ["left", "right", "top"]:
        ax1.spines[spine].set_visible(False)

    plt.savefig(path, bbox_inches="tight")
    plt.close()


def rounds(data, path, model):
    total_time = data[-1][0] if data[-1] else None

    raw_data = data[:-1]
    operators = [item[0] for item in raw_data]
    gen_rounds = [item[1] for item in raw_data]
    repair_rounds = [item[2] for item in raw_data]
    ce_counts = [item[3] for item in raw_data]
    times = [item[4] for item in raw_data]
    success_flags = [item[5] for item in raw_data]

    x = np.arange(len(operators)) * 1.4
    bar_width = 0.3

    fig, ax1 = plt.subplots(figsize=(13, 6))
    # fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax1.set_facecolor("#f0f0f0")
    ax1.grid(True, axis="y", color="white", linewidth=1.2)
    ax1.set_axisbelow(True)

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

    ax1.set_ylabel("Numbers", fontsize=10, labelpad=10, rotation=0, loc="top")
    ax1.yaxis.set_label_coords(0.037, 1.02)
    # ax1.set_xlabel("Operators")
    ax1.set_xticks(x)
    ax1.set_xticklabels(operators, rotation=45, ha="right")

    """
    ax2 = ax1.twinx()
    line = ax2.plot(
        x, times, color="#4b6f8d", marker="o", label="Time (s)", linewidth=2
    )
    ax2.set_ylabel("Time (s)")
    """

    lines_labels = ax1.get_legend_handles_labels()
    # lines_labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(
    #     lines_labels[0] + lines_labels2[0],
    #     lines_labels[1] + lines_labels2[1],
    #     loc="upper right",
    # )

    extra_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✓$",
            markersize=12,
            linestyle="None",
            label="Sound",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✗$",
            markersize=12,
            linestyle="None",
            label="Unsound",
        ),
    ]

    handles, labels = ax1.get_legend_handles_labels()

    ax1.legend(
        handles + extra_handles,
        labels + ["Sound Generation", "Unsound Generation"],
        loc="upper right",
        fontsize=12,
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
            fontsize=8,
        )

    for i in range(len(operators)):
        max_height = max(gen_rounds[i], repair_rounds[i], ce_counts[i])
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax1.text(x[i], max_height + 5, symbol, ha="center", va="bottom", fontsize=12)

    # plt.title(
    #    "LLM Generation Rounds, LLM Repair Rounds, Counterexamples, and Time per Operator"
    # )

    # plt.tight_layout()

    # if total_time is not None:
    #     fig.text(
    #         0.5,
    #         0.02,
    #         f"Total runtime for model {model}: {total_time:.2f} seconds",
    #         ha="center",
    #         va="bottom",
    #         fontsize=10,
    #     )

    for spine in ["left", "right", "top"]:
        ax1.spines[spine].set_visible(False)

    plt.savefig(path, bbox_inches="tight")


def rounds_gen_repair(data, path, model):  # for ablation study
    total_time = data[-1][0] if data[-1] else None

    raw_data = data[:-1]
    operators = [item[0] for item in raw_data]
    gen_rounds = [item[1] for item in raw_data]
    repair_rounds = [item[2] for item in raw_data]
    ce_counts = [item[3] for item in raw_data]
    times = [item[4] for item in raw_data]
    success_flags = [item[5] for item in raw_data]

    x = np.arange(len(operators)) * 1.4
    bar_width = 0.3

    fig, ax1 = plt.subplots(figsize=(13, 6))
    # fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax1.set_facecolor("#f0f0f0")
    ax1.grid(True, axis="y", color="white", linewidth=1.2)
    ax1.set_axisbelow(True)

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

    ax1.set_ylabel("Numbers", fontsize=10, labelpad=10, rotation=0, loc="top")
    ax1.yaxis.set_label_coords(0.026, 1.02)
    # ax1.set_xlabel("Operators")
    ax1.set_xticks(x)
    ax1.set_xticklabels(operators, rotation=45, ha="right")

    """
    ax2 = ax1.twinx()
    line = ax2.plot(
        x, times, color="#4b6f8d", marker="o", label="Time (s)", linewidth=2
    )
    ax2.set_ylabel("Time (s)")
    """

    extra_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✓$",
            markersize=12,
            linestyle="None",
            label="Sound",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✗$",
            markersize=12,
            linestyle="None",
            label="Unsound",
        ),
    ]

    # handles, labels = ax1.get_legend_handles_labels()

    # ax1.legend(
    #     handles + extra_handles,
    #     labels + ["Sound Generation", "Unsound Generation"],
    #     loc="upper right",
    #     fontsize=12,
    # )

    max_height = max(
        max(gen_rounds, default=0),
        max(repair_rounds, default=0),
        max(ce_counts, default=0),
    )

    ax1.set_yticks(np.arange(0, max_height + 15, 35))

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for i in range(len(operators)):
        max_height = max(gen_rounds[i], repair_rounds[i], ce_counts[i])
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax1.text(x[i], max_height + 15, symbol, ha="center", va="bottom", fontsize=12)

    for spine in ["left", "right", "top"]:
        ax1.spines[spine].set_visible(False)

    plt.savefig(path, bbox_inches="tight")


def draw_all(statistic, dir):
    for model_name, data in statistic.items():
        path = os.path.join(dir, f"{model_name}.pdf")
        rounds(data, path, model_name)


def draw_cost_curve(op, costs, output_dir):
    if not costs:
        return

    rounds = list(range(1, len(costs) + 1))

    chain_idxs = [0]
    next_idx = 0
    for j in range(1, len(costs)):
        if costs[j] < costs[next_idx] - 0.0001:
            next_idx = j
            chain_idxs.append(next_idx)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{op}_cost_curve.pdf")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)

    plt.scatter(rounds, costs)

    chain_x = [rounds[k] for k in chain_idxs]
    chain_y = [costs[k] for k in chain_idxs]
    plt.plot(chain_x, chain_y, marker="o", linewidth=1)

    for x, y in zip(chain_x, chain_y):
        offset = 8
        va = "bottom"
        if y == max(costs):
            offset = -12
            va = "top"
        plt.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va=va,
            fontsize=12,
        )

    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.xticks(rounds)
    # plt.xlabel("Rounds",fontsize=12)
    ax.set_ylabel("Cost Function", fontsize=12, labelpad=12, rotation=0, loc="top")
    ax.yaxis.set_label_coords(0.155, 1.02)
    plt.xticks(rounds, fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f"{op} — Cost Function vs. Rounds")
    plt.grid(True, linestyle="-", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def rq1(outfile="rq1.pdf", annotate=True):

    data_map = {
        "GPT-5": [1, 0, 0, True],
        "GPT-4o": [1, 1, 3, True],
        "Llama4-Maverick": [28, 39, 16, True],
        "Claude-Opus-4": [40, 3, 9, True],
    }

    model_names = list(data_map.keys())
    gen_vals = [data_map[m][0] for m in model_names]
    repair_vals = [data_map[m][1] for m in model_names]
    ce_vals = [data_map[m][2] for m in model_names]
    success_flags = [data_map[m][3] for m in model_names]

    # bkg
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    x = np.arange(len(model_names))
    bar_width = 0.22

    b1 = ax.bar(
        x - bar_width,
        gen_vals,
        width=bar_width,
        label="LLM Gen Rounds",
        color="#f7b7a3",
    )
    b2 = ax.bar(
        x,
        repair_vals,
        width=bar_width,
        label="LLM Repair Rounds",
        color="#f6d5e3",
    )
    b3 = ax.bar(
        x + bar_width,
        ce_vals,
        width=bar_width,
        label="Counterexamples",
        color="#7e9ce0",
    )

    ax.set_ylabel("Numbers", fontsize=14, labelpad=10, rotation=0, loc="top")
    ax.yaxis.set_label_coords(0.075, 1.01)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, ha="center")

    ax.tick_params(axis="x", labelsize=13)

    ax.tick_params(axis="y", labelsize=14)

    extra_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✓$",
            markersize=12,
            linestyle="None",
            label="Sound",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✗$",
            markersize=12,
            linestyle="None",
            label="Unsound",
        ),
    ]

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles + extra_handles,
        labels + ["Sound Generation", "Unsound Generation"],
        loc="upper left",
        fontsize=14,
    )

    ymax = (
        max(max(gen_vals), max(repair_vals), max(ce_vals))
        if len(model_names) > 0
        else 0
    )
    ax.set_ylim(0, ymax * 1.25 + 1)

    if annotate:
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(0.02 * ymax, 0.2),
                    f"{int(h)}" if float(h).is_integer() else f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    for i in range(len(model_names)):
        max_height = max(gen_vals[i], repair_vals[i], ce_vals[i])
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax.text(
            x[i],
            max_height + max(0.08 * ymax, 1.0),
            symbol,
            ha="center",
            va="bottom",
            fontsize=20,
            # color="green" if success_flags[i] else "red",
            fontweight="bold",
        )

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def rq3(outfile="rq3.pdf", annotate=True):

    data_map = {
        "Gelu": [10, 3, 9, True],
        "Elu": [7, 10, 4, True],
        "Sigmoid": [1, 0, 0, True],
    }

    model_names = list(data_map.keys())
    gen_vals = [data_map[m][0] for m in model_names]
    repair_vals = [data_map[m][1] for m in model_names]
    ce_vals = [data_map[m][2] for m in model_names]
    success_flags = [data_map[m][3] for m in model_names]

    # bkg
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    x = np.arange(len(model_names))
    bar_width = 0.22

    # fig, ax = plt.subplots(figsize=(8, 5))
    # plt.subplots_adjust(bottom=0.18)

    b1 = ax.bar(
        x - bar_width,
        gen_vals,
        width=bar_width,
        label="LLM Gen Rounds",
        color="#f7b7a3",
    )
    b2 = ax.bar(
        x,
        repair_vals,
        width=bar_width,
        label="LLM Repair Rounds",
        color="#f6d5e3",
    )
    b3 = ax.bar(
        x + bar_width,
        ce_vals,
        width=bar_width,
        label="Counterexamples",
        color="#7e9ce0",
    )

    ax.set_ylabel("Numbers", fontsize=14, labelpad=10, rotation=0, loc="top")
    ax.yaxis.set_label_coords(0.075, 1.01)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0, ha="center")

    ax.tick_params(axis="x", labelsize=13)

    ax.tick_params(axis="y", labelsize=14)

    extra_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✓$",
            markersize=12,
            linestyle="None",
            label="Sound",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker=r"$✗$",
            markersize=12,
            linestyle="None",
            label="Unsound",
        ),
    ]

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles + extra_handles,
        labels + ["Sound Generation", "Unsound Generation"],
        loc="upper right",
        fontsize=14,
    )

    ymax = (
        max(max(gen_vals), max(repair_vals), max(ce_vals))
        if len(model_names) > 0
        else 0
    )
    ax.set_ylim(0, ymax * 1.25 + 1)

    if annotate:
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(0.02 * ymax, 0.2),
                    f"{int(h)}" if float(h).is_integer() else f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    for i in range(len(model_names)):
        max_height = max(gen_vals[i], repair_vals[i], ce_vals[i])
        symbol = "\u2713" if success_flags[i] else "\u2717"
        ax.text(
            x[i],
            max_height + max(0.08 * ymax, 1.0),
            symbol,
            ha="center",
            va="bottom",
            fontsize=20,
            # color="green" if success_flags[i] else "red",
            fontweight="bold",
        )

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


# HardSigmoid function
def hardsigmoid_curve():
    def hardsigmoid(x):
        return np.clip((x + 3) / 6, 0, 1)

    x = np.linspace(-9, 9, 400)
    y = hardsigmoid(x)

    # Plot style setup
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Background & grid
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2)

    # Draw main line (emphasized, warm color)
    ax.plot(x, y, color="#c05020", linewidth=3)

    # Labels
    ax.set_xlabel("x", fontsize=12, labelpad=6, loc="center")
    ax.set_ylabel("y", fontsize=12, labelpad=10, rotation=0, ha="center")
    ax.yaxis.set_label_coords(-0.045, 1.02)  # (x_offset, y_offset)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    # Remove legend
    ax.text(1, 0.45, "HardSigmoid(x)", color="#c05020", fontsize=12, weight="bold")

    plt.savefig("hardsigmoid.pdf", bbox_inches="tight")
    # Tight layout
    plt.tight_layout()
    plt.show()


def hardsigmoid_incorrect():
    def hardsigmoid(x):
        return np.clip((x + 3) / 6, 0, 1)

    x = np.linspace(-6, 6, 600)
    y = hardsigmoid(x)

    # two bounds
    # y = (x + 4) / 8
    sec = (x + 4.0) / 8.0
    sigma = (x + 3) / 6
    lower = sec
    upper = sec

    # bkg
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2, zorder=0)

    ax.plot(x, y, color="#c05020", linewidth=2.5, label="HardSigmoid(x)")

    ax.plot(x, lower, color="blue", linewidth=3, label="Lower: sec((l_i,0),(u_i,1))")
    ax.plot(
        x,
        upper,
        color="green",
        # linestyle="dashdot",
        linewidth=1.8,
        label="Upper: sec((l_i,0),(u_i,1))",
    )

    mask = (x >= -4) & (x <= 0)
    base = np.where(x <= -3, 0.0, sigma)
    lower = np.minimum(base, sec)
    upper = np.maximum(base, sec)

    ax.fill_between(
        x[mask], lower[mask], upper[mask], color="#442929", alpha=0.4, zorder=3
    )

    mask2 = (x >= 0) & (x <= 4)
    base = np.where(x <= 3, sigma, 1.0)
    lower2 = np.minimum(base, sec)
    upper2 = np.maximum(base, sec)
    ax.fill_between(
        x[mask2], lower2[mask2], upper2[mask2], color="#442929", alpha=0.4, zorder=3
    )

    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.2, 1.25)

    ax.set_xticks([])

    # ax.set_yticks([])
    ax.tick_params(axis="y", labelsize=20)

    ax.set_xlabel("x", fontsize=19, loc="right")
    # ax.set_ylabel("y", fontsize=14,loc="top")
    ax.set_ylabel("y", fontsize=19, labelpad=10, rotation=0, loc="top")
    ax.yaxis.set_label_coords(-0.08, 1.06)

    # l and u
    ax.plot([-4, -4], [0, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(-4, -0.25, r"$\ell_i$", ha="center", va="top", fontsize=20)

    # (4,1) u
    ax.plot([4, 4], [1, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(4, -0.25, r"$u_i$", ha="center", va="top", fontsize=20)

    ax.plot([-3, -3], [0, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(-3, -0.25, r"$-3$", ha="center", va="top", fontsize=20)
    ax.plot([3, 3], [1, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(3, -0.25, r"$3$", ha="center", va="top", fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        fontsize=17,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
    )

    plt.tight_layout()
    plt.savefig("hardsigmoid_incorrect.pdf", bbox_inches="tight")
    plt.show()


def hardsigmoid_correct():
    def hardsigmoid(x):
        return np.clip((x + 3) / 6, 0, 1)

    x = np.linspace(-6, 6, 600)
    y = hardsigmoid(x)

    sec1 = (x + 4.0) / 7.0
    sec2 = (x + 3.0) / 7.0
    sigma = (x + 3) / 6
    lower = sec2
    upper = sec1

    # bkg
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.2, zorder=0)

    ax.plot(x, y, color="#c05020", linewidth=2.5, label="HardSigmoid(x)")

    ax.plot(x, lower, color="blue", linewidth=1.8, label="Lower: sec((-3,0),(u_i,1))")
    ax.plot(
        x,
        upper,
        color="green",
        # linestyle="dash",
        linewidth=1.8,
        label="Upper: sec((l_i,0),(3,1))",
    )

    mask = (x >= -4) & (x <= 4)
    base1 = np.where(x <= -3, 0.0, sec2)
    base2 = np.where(x >= 3, 1.0, sec1)
    # lower2 = np.minimum(base1, sec2)
    # upper2= np.maximum(base2, sec1)
    ax.fill_between(
        x[mask], base1[mask], base2[mask], color="#442929", alpha=0.4, zorder=3
    )

    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.2, 1.25)

    ax.set_xticks([])

    # ax.set_yticks([])
    ax.tick_params(axis="y", labelsize=20)

    ax.set_xlabel("x", fontsize=19, loc="right")
    # ax.set_ylabel("y", fontsize=14,loc="top")
    ax.set_ylabel("y", fontsize=19, labelpad=10, rotation=0, loc="top")
    ax.yaxis.set_label_coords(-0.08, 1.06)

    # l and u
    ax.plot([-4, -4], [0, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(-4, -0.25, r"$\ell_i$", ha="center", va="top", fontsize=20)

    # (4,1) u
    ax.plot([4, 4], [1, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(4, -0.25, r"$u_i$", ha="center", va="top", fontsize=20)

    ax.plot([-3, -3], [0, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(-3, -0.25, r"$-3$", ha="center", va="top", fontsize=20)
    ax.plot([3, 3], [1, -0.2], linestyle="--", color="black", linewidth=1)
    ax.text(3, -0.25, r"$3$", ha="center", va="top", fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        fontsize=17,
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
    )

    plt.tight_layout()
    plt.savefig("hardsigmoid_correct.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    rq1_claude_hardsigmoid_cost_curve = [
        1.1667,
        0.08109292760935044,
        0.6667,
        0.0833,
        0.1667,
        16.1667,
        0.5,
        0.3333,
        1.1667,
        0,
    ]

    rq1_llama_hardsigmoid_cost_curve = [
        2.524,
        0.173087721909493,
        0.25142700890640024,
        0.17718345911421143,
        0.173087721909493,
        0.06720,
        0.03600467232176883,
        0.03461754438189985,
        0.2722222222222222,
        0.2913643318809905,
        0.05294622854807017,
        0.053266363304609626,
        0.05294622854807017,
        0.09410754290385964,
        0.05294622854807017,
        0.03600467232176883,
        0,
    ]

    rq1_gpt4o_hardsigmoid_cost_curve = [
        0.22856178724088122,
        0.74328758395739752,
        0.13749257934790754,
        0,
    ]

    llama_deeppoly = [
        ["Abs", 1, 0, 0, 4.291302442550659, True],
        ["Affine", 15, 0, 0, 46.63023924827576, False],
        ["Avgpool", 2, 9, 0, 1725.8876268863678, True],
        ["HardSigmoid", 28, 39, 16, 377.522705078125, True],
        ["HardSwish", 40, 6, 10, 2342.763063430786, False],
        ["HardTanh", 75, 14, 15, 1080.2573719024658, True],
        ["Maxpool", 15, 34, 0, 4772.505630970001, False],
        ["Minpool", 15, 47, 0, 8873.733200073242, False],
        ["Neuron_add", 1, 0, 0, 3.927150249481201, True],
        ["Neuron_max", 2, 4, 1, 1501.0491557121277, True],
        ["Neuron_min", 1, 1, 0, 265.8800919055939, True],
        ["Neuron_mult", 2, 7, 2, 2242.203906774521, True],
        ["Relu", 1, 0, 0, 4.36462926864624, True],
        ["Relu6", 1, 1, 0, 309.66770696640015, True],
        [33886.721366882324],
    ]

    llama_deeppoly_wo_feedback = [
        ["Abs", 1, 0, 0, 4.500029563903809, True],
        ["Affine", 80, 0, 0, 210.5168011188507, False],
        ["Avgpool", 5, 15, 1, 16.952557802200317, True],
        ["HardSigmoid", 80, 552, 0, 6920.944112539291, False],
        ["HardSwish", 80, 447, 0, 10259.599481344223, False],
        ["HardTanh", 80, 414, 0, 9960.046389341354, False],
        ["Maxpool", 80, 183, 0, 2819.499320745468, False],
        ["Minpool", 80, 630, 0, 665.0382826328278, False],
        ["Neuron_add", 1, 0, 0, 2.8666207790374756, True],
        ["Neuron_max", 80, 339, 5, 433.0541205406189, False],
        ["Neuron_min", 80, 540, 0, 266.23403000831604, False],
        ["Neuron_mult", 80, 18, 0, 671.9914238452911, False],
        ["Relu", 1, 0, 0, 5.813177108764648, True],
        ["Relu6", 80, 99, 2, 576.1142089366913, False],
        [32813.18037867546],
    ]

    llama_deeppoly_wo_feedback_w_repair = [
        ["Abs", 1, 0, 0, 4.999667644500732, True],
        ["Affine", 80, 0, 0, 268.98456501960754, False],
        ["Avgpool", 1, 0, 0, 2.7647504806518555, True],
        ["HardSigmoid", 80, 522, 0, 7097.016361236572, False],
        ["HardSwish", 80, 414, 0, 11825.581152200699, False],
        ["HardTanh", 80, 360, 0, 3860.305430650711, False],
        ["Maxpool", 80, 89, 0, 742.1072971820831, False],
        ["Minpool", 80, 519, 0, 1999.8638792037964, False],
        ["Neuron_add", 1, 0, 0, 3.2583985328674316, True],
        ["Neuron_max", 80, 305, 3, 3243.4616799354553, False],
        ["Neuron_min", 80, 391, 0, 2469.9393882751465, False],
        ["Neuron_mult", 80, 18, 0, 813.2067356109619, False],
        ["Relu", 1, 0, 0, 8.5110182762146, True],
        ["Relu6", 80, 77, 1, 2654.239928007126, False],
        [34994.250092983246],
    ]

    # rounds(llama_deeppoly, "./llama_deeppoly.pdf", "llama4-maverick")
    rounds_gen_only(
        llama_deeppoly_wo_feedback,
        "./llama_deeppoly_wo_feedback.pdf",
        "llama4-maverick",
    )
    rounds_gen_repair(
        llama_deeppoly_wo_feedback_w_repair,
        "./llama_deeppoly_wo_feedback_w_repair.pdf",
        "llama4-maverick",
    )
    # rounds(gpt5_deeppoly, "./gpt5_deeppoly.pdf", "gpt5")
    # rounds(claude_deeppoly, "./claude_deeppoly.pdf", "claude")
    # rounds(gpt5_deepz, "./gpt5_deepz.pdf", "gpt5")
    # rounds_gen_only(llama_deeppoly_wo_feedback, "./llama_wo_feedback.pdf", "llama4-maverick")

    # RQ1:
    # rq1()
# draw_cost_curve("HardSigmoid", rq1_gpt4o_hardsigmoid_cost_curve, "./")
# draw_cost_curve("HardSigmoid", rq1_llama_hardsigmoid_cost_curve, "./")
# draw_cost_curve("HardSigmoid", rq1_claude_hardsigmoid_cost_curve, "./")

# RQ3:
# rq3()

# overview:
# hardsigmoid_incorrect()
# hardsigmoid_correct()
