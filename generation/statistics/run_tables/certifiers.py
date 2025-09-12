import matplotlib.pyplot as plt

# Operator list
operators = [
    "Abs",
    "Relu",
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
    "Add",
    "Relu6",
]

deepseek_results = {
    "DeepPoly": {
        "Abs": True,
        "Relu": True,
        "Affine": True,
        "Avgpool": True,
        "HardSigmoid": True,
        "HardSwish": True,
        "HardTanh": True,
        "Max": False,
        "Maxpool": True,
        "Min": False,
        "Minpool": False,
        "Mult": False,
        "Add": False,
        "Relu6": False,
    },
    "IBP": {
        "Abs": True,
        "Relu": True,
        "Affine": True,
        "Avgpool": False,
        "HardSigmoid": True,
        "HardSwish": False,
        "HardTanh": True,
        "Max": False,
        "Maxpool": False,
        "Min": False,
        "Minpool": True,
        "Mult": False,
        "Add": False,
        "Relu6": False,
    },
    "DeepZ": {
        "Abs": True,
        "Relu": True,
        "Affine": True,
        "Avgpool": False,
        "HardSigmoid": False,
        "HardSwish": False,
        "HardTanh": False,
        "Max": False,
        "Maxpool": True,
        "Min": False,
        "Minpool": False,
        "Mult": False,
        "Add": False,
        "Relu6": False,
    },
}

total_time = {
    "DeepPoly": 1325,
    "IBP": 1360,
    "DeepZ": 1488,
}


table_data = []
for op in operators:
    row = [op]
    for certifier in ["DeepPoly", "IBP", "DeepZ"]:
        success = deepseek_results[certifier].get(op, False)
        row.append("\u2713" if success else "")
    table_data.append(row)

time_row = ["Total Time (s)"] + [
    str(total_time[cert]) for cert in ["DeepPoly", "IBP", "DeepZ"]
]
table_data.append(time_row)

columns = ["Operator", "DeepPoly", "IBP", "DeepZ"]


fig, ax = plt.subplots(figsize=(6, 0.4 * len(operators) + 1))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")


table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)


plt.tight_layout()
plt.savefig("pics/deepseek_certifier_success_table.png", dpi=300)
plt.close()
