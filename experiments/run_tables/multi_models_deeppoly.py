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


def draw_nine(statistics, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(statistics.items()):
        rounds_subplot(axes[idx], data, model_name)

    # Remove unused subplots if <9
    for ax in axes[len(statistics) :]:
        ax.axis("off")

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)

    plt.suptitle("LLM Rounds, Counterexamples, and Time per Operator", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.show()


def draw_eight(statistics, output_path):
    num_models = len(statistics)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(statistics.items()):
        rounds_subplot(axes[idx], data, model_name)

    for ax in axes[num_models:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)

    plt.suptitle(
        "LLM Rounds, Counterexamples, and Time per Operator (Certifier: DeepPoly)",
        fontsize=14,
        y=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Suppose you have 9 datasets in a dict
    statistics = {
        "Deepseek-r1-v1:0": [
            ["Abs", 1, 0, 0, 33.41599249839783, True],
            ["Affine", 1, 0, 0, 12.662463665008545, True],
            ["Avgpool", 10, 0, 0, 689.1402835845947, False],
            ["HardSigmoid", 10, 0, 0, 599.6716084480286, False],
            ["HardSwish", 10, 0, 0, 599.9701881408691, False],
            ["HardTanh", 10, 0, 0, 599.991673707962, False],
            ["Maxpool", 10, 3, 0, 1679.9965977668762, False],
            ["Minpool", 10, 0, 0, 600.062656879425, False],
            ["Neuron_add", 1, 0, 0, 60.26461720466614, True],
            ["Neuron_max", 10, 3, 0, 719.7148790359497, False],
            ["Neuron_min", 10, 0, 0, 600.0296392440796, False],
            ["Neuron_mult", 10, 3, 0, 780.0147662162781, False],
            ["Relu", 1, 0, 0, 60.806447982788086, True],
            ["Relu6", 10, 0, 0, 599.1907169818878, False],
            [7634.948748111725],
        ],
        "gpt-4o": [
            ["Abs", 1, 0, 0, 59.05008816719055, True],
            ["Affine", 1, 2, 0, 120.81806182861328, True],
            ["Avgpool", 3, 1, 0, 479.6388111114502, True],
            ["HardSigmoid", 10, 10, 1, 1560.6130983829498, False],
            ["HardSwish", 10, 9, 2, 962.2697412967682, False],
            ["HardTanh", 1, 1, 1, 123.34995412826538, True],
            ["Maxpool", 7, 25, 0, 1073.7991869449615, True],
            ["Minpool", 7, 28, 0, 1140.0606484413147, True],
            ["Neuron_add", 1, 0, 0, 6.775387763977051, True],
            ["Neuron_max", 1, 2, 0, 63.304216384887695, True],
            ["Neuron_min", 10, 36, 3, 1494.6577343940735, True],
            ["Neuron_mult", 10, 57, 0, 1555.3929617404938, False],
            ["Relu", 1, 0, 0, 7.352106332778931, True],
            ["Relu6", 10, 12, 6, 1672.7748591899872, False],
            [10319.883068561554],
        ],
        "gpt-5": [
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
        "Claude-opus-4-1-20250805-v1:0": [
            ["Abs", 1, 0, 1, 59.773683071136475, True],
            ["Affine", 1, 0, 0, 58.578789472579956, True],
            ["Avgpool", 7, 25, 0, 2400.099935531616, True],
            ["HardSigmoid", 10, 0, 3, 1679.6386258602142, False],
            ["HardSwish", 10, 0, 3, 904.8073153495789, False],
            ["HardTanh", 10, 0, 3, 862.3565874099731, False],
            ["Maxpool", 10, 3, 0, 3872.6559710502625, False],
            ["Minpool", 10, 0, 0, 2400.0256700515747, False],
            ["Neuron_add", 1, 0, 0, 60.6185781955719, True],
            ["Neuron_max", 1, 1, 0, 120.71896004676819, True],
            ["Neuron_min", 10, 4, 5, 2038.7003786563873, False],
            ["Neuron_mult", 10, 46, 0, 5705.430655479431, False],
            ["Relu", 1, 0, 0, 115.28929543495178, True],
            ["Relu6", 10, 0, 2, 1384.33265376091, False],
            [21663.060158729553],
        ],
        "Claude-opus-4-20250514-v1:0": [
            ["Abs", 4, 0, 4, 475.5917546749115, True],
            ["Affine", 1, 0, 0, 59.626365184783936, True],
            ["Avgpool", 10, 51, 0, 4199.267009735107, False],
            ["HardSigmoid", 10, 0, 2, 3421.704955101013, False],
            ["HardSwish", 10, 0, 2, 1858.6576886177063, False],
            ["HardTanh", 10, 0, 3, 1319.5205919742584, False],
            ["Maxpool", 10, 6, 0, 4860.126341104507, False],
            ["Minpool", 10, 3, 0, 4920.03166270256, False],
            ["Neuron_add", 1, 0, 0, 60.37423849105835, True],
            ["Neuron_max", 9, 1, 1, 1080.5177443027496, True],
            ["Neuron_min", 10, 0, 5, 1979.5036494731903, False],
            ["Neuron_mult", 10, 57, 0, 4919.809273719788, False],
            ["Relu", 1, 0, 0, 60.61888003349304, True],
            ["Relu6", 10, 0, 3, 1440.821168422699, False],
            [30656.199625730515],
        ],
        "Claude-sonnet-4-20250514-v1:0": [
            ["Abs", 1, 0, 0, 42.62267327308655, True],
            ["Affine", 1, 0, 0, 59.93818283081055, True],
            ["Avgpool", 10, 0, 0, 600.3020262718201, False],
            ["HardSigmoid", 10, 0, 5, 1203.0894885063171, False],
            ["HardSwish", 10, 0, 4, 1559.1923146247864, False],
            ["HardTanh", 10, 0, 0, 1977.8736746311188, False],
            ["Maxpool", 10, 90, 0, 3479.1033368110657, False],
            ["Minpool", 10, 84, 0, 3022.629314184189, False],
            ["Neuron_add", 1, 0, 0, 17.74017834663391, True],
            ["Neuron_max", 10, 39, 3, 2642.290125846863, False],
            ["Neuron_min", 10, 28, 3, 3134.5421459674835, False],
            ["Neuron_mult", 10, 43, 0, 5283.858697891235, False],
            ["Relu", 1, 0, 0, 59.852705240249634, True],
            ["Relu6", 9, 0, 4, 1320.3963994979858, True],
            [24403.457837820053],
        ],
        "Llama3-3-70b-instruct-v1:0": [
            ["Abs", 2, 2, 2, 28.430330753326416, True],
            ["Affine", 1, 0, 0, 3.7514688968658447, True],
            ["Avgpool", 10, 6, 0, 176.93466091156006, False],
            ["HardSigmoid", 10, 0, 2, 462.5687539577484, False],
            ["HardSwish", 10, 0, 4, 530.046600818634, False],
            ["HardTanh", 10, 1, 7, 300.8974447250366, False],
            ["Maxpool", 10, 44, 0, 272.5759389400482, False],
            ["Minpool", 10, 47, 0, 294.2355501651764, False],
            ["Neuron_add", 1, 0, 0, 30.852378845214844, True],
            ["Neuron_max", 1, 1, 0, 9.296123266220093, True],
            ["Neuron_min", 1, 1, 0, 11.558663368225098, True],
            ["Neuron_mult", 10, 36, 0, 586.0417859554291, False],
            ["Relu", 1, 0, 0, 3.597078561782837, True],
            ["Relu6", 10, 37, 3, 372.71810245513916, False],
            [3083.513258934021],
        ],
        "Llama4-maverick-17b-instruct-v1:0": [
            ["Abs", 1, 0, 1, 57.41964387893677, True],
            ["Affine", 1, 0, 0, 3.0262932777404785, True],
            ["Avgpool", 10, 30, 0, 424.94965291023254, False],
            ["HardSigmoid", 10, 0, 3, 481.3030390739441, False],
            ["HardSwish", 10, 0, 2, 413.0324373245239, False],
            ["HardTanh", 2, 0, 1, 57.47521901130676, True],
            ["Maxpool", 10, 21, 0, 367.25705099105835, False],
            ["Minpool", 10, 8, 0, 77.60117173194885, False],
            ["Neuron_add", 1, 0, 0, 2.861654281616211, True],
            ["Neuron_max", 1, 1, 0, 6.0658721923828125, True],
            ["Neuron_min", 1, 1, 0, 4.799383163452148, True],
            ["Neuron_mult", 10, 9, 0, 96.05530524253845, False],
            ["Relu", 1, 0, 0, 3.8151185512542725, True],
            ["Relu6", 10, 0, 2, 144.57383155822754, False],
            [2140.2547714710236],
        ],
    }
    draw_eight(statistics, "pics/rmulti_models_deeppoly.png")
