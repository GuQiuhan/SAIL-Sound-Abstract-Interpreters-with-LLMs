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
        loc="upper right",
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

    plt.title(
        "LLM Generation Rounds, LLM Repair Rounds, Counterexamples, and Time per Operator"
    )

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

    plt.savefig(path, bbox_inches="tight")


def draw_all(statistic, dir):
    for model_name, data in statistic.items():
        path = os.path.join(dir, f"{model_name}.pdf")
        rounds(data, path, model_name)


def draw_cost_curve(op, costs, output_dir):
    if not costs:
        return

    rounds = list(range(1, len(costs) + 1))

    chain_idxs = []
    i = 0
    chain_idxs.append(i)
    next_idx = 0
    chain_idxs.append(next_idx)
    for j in range(1, len(costs)):
        if costs[j] < costs[next_idx] - 0.0001:
            next_idx = j
            chain_idxs.append(next_idx)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{op}_cost_curve.pdf")

    plt.subplots(figsize=(6, 4), dpi=150)

    plt.scatter(rounds, costs)

    chain_x = [rounds[k] for k in chain_idxs]
    chain_y = [costs[k] for k in chain_idxs]
    plt.plot(chain_x, chain_y, marker="o", linewidth=1)

    # for x, y in zip(rounds, costs):
    #    plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
    #                 xytext=(0, 8), ha='center', fontsize=9)

    for x, y in zip(rounds, costs):
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
            fontsize=6,
        )

    plt.xticks(rounds)
    plt.xlabel("")
    plt.ylabel("Cost Function")
    # plt.title(f"{op} — Cost Function vs. Rounds")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def rq1(outfile="rq1.pdf", annotate=True):

    data_map = {
        "GPT-5": [1, 0, 0, True],
        "GPT-4o": [6, 0, 5, True],
        "Llama4-maverick": [28, 39, 16, True],
        "Claude-opus-4": [40, 3, 9, True],
    }

    model_names = list(data_map.keys())
    gen_vals = [data_map[m][0] for m in model_names]
    repair_vals = [data_map[m][1] for m in model_names]
    ce_vals = [data_map[m][2] for m in model_names]
    success_flags = [data_map[m][3] for m in model_names]

    x = np.arange(len(model_names))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)

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

    ax.set_ylabel("Number of Rounds / Counterexamples")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")

    ax.legend(loc="upper left")

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
                    fontsize=8,
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
            fontsize=12,
            # color="green" if success_flags[i] else "red",
            fontweight="bold",
        )

    fig.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)


def rq1(outfile="rq2.pdf", annotate=True):

    data_map = {
        "Sigmoid": [1, 0, 0, True],
        "Elu": [2, 0, 5, True],
        "Selu": [28, 39, 16, True],
        "Gelu": [40, 3, 9, True],
    }

    model_names = list(data_map.keys())
    gen_vals = [data_map[m][0] for m in model_names]
    repair_vals = [data_map[m][1] for m in model_names]
    ce_vals = [data_map[m][2] for m in model_names]
    success_flags = [data_map[m][3] for m in model_names]

    x = np.arange(len(model_names))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)

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

    ax.set_ylabel("Number of Rounds / Counterexamples")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")

    ax.legend(loc="upper left")

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
                    fontsize=8,
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
            fontsize=12,
            # color="green" if success_flags[i] else "red",
            fontweight="bold",
        )

    fig.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)
