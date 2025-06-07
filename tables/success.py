import matplotlib.pyplot as plt
import pandas as pd


model_names = ["DeepSeek-V2-Lite", "Llama-3.3-70B-Instruct", "Gpt-4o"]


results = [
    [  # DeepSeek
        ["Abs", 10, True], ["Add", 79, False], ["Affine", 45, True], ["Avgpool", 38, True],
        ["HardSigmoid", 84, True], ["HardSwish", 34, True], ["HardTanh", 64, True],
        ["Max", 70, False], ["Maxpool", 61, True], ["Min", 100, False], ["Minpool", 80, False],
        ["Mult", 67, False], ["Relu", 7, True], ["Relu6", 79, False]
    ],
    [  # Llama
        ["Abs", 69, True], ["Add", 814, False], ["Affine", 510, False], ["Avgpool", 547, False],
        ["HardSigmoid", 1107, False], ["HardSwish", 1625, False], ["HardTanh", 811, False],
        ["Max", 857, False], ["Maxpool", 954, False], ["Min", 1553, False], ["Minpool", 1140, False],
        ["Mult", 1414, False], ["Relu", 871, False], ["Relu6", 871, False]
    ],
    [  # Gpt-4o
        ["Abs", 0, True], ["Add", 0, False], ["Affine", 0, True], ["Avgpool", 0, False],
        ["HardSigmoid", 0, False], ["HardSwish", 0, False], ["HardTanh", 0, False],
        ["Max", 0, False], ["Maxpool", 0, False], ["Min", 0, False], ["Minpool", 0, False],
        ["Mult", 0, False], ["Relu", 0, True], ["Relu6", 0, False]
    ]
]


operator_names = [op[0] for op in results[0]]


success_matrix = []
for i in range(len(operator_names)):
    row = []
    for model_result in results:
        row.append("\u2713" if model_result[i][2] else "")
    success_matrix.append(row)


df = pd.DataFrame(success_matrix, columns=model_names, index=operator_names)


fig, ax = plt.subplots(figsize=(9, len(operator_names) * 0.5 + 1))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    loc='center',
    cellLoc='center',
    colLoc='center'
)
table.scale(1, 1.5)
plt.title("Operator Generation Success Table", fontsize=14)


plt.savefig("pics/success_table.png", dpi=300, bbox_inches='tight')
plt.show()
