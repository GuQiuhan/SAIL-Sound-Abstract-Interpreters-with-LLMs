import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times"]

data = {
    "mnist_relu_3_50.onnx": [
        "MNIST",
        "3x50",
        "Standard",
        "FCN_Relu",
        3,
        [
            [0.005, 100, 0.5099999904632568, 0.1450357437133789],
            [0.0005, 100, 0.949999988079071, 0.10021162033081055],
            [0.0001, 100, 0.9700000286102295, 0.1415255069732666],
        ],
    ],
    "mnist_relu_3_100.onnx": [
        "MNIST",
        "3x100",
        "Standard",
        "FCN_Relu",
        3,
        [
            [0.005, 100, 0.10999999940395355, 0.10600137710571289],
            [0.0005, 100, 0.9399999976158142, 0.09951305389404297],
            [0.0001, 100, 0.9800000190734863, 0.10778522491455078],
        ],
    ],
    "mnist_relu_5_100.onnx": [
        "MNIST",
        "3x100",
        "DiffAI",
        "FCN_Relu",
        6,
        [
            [0.005, 100, 0.0, 0.24838042259216309],
            [0.0005, 100, 0.029999999329447746, 0.23444795608520508],
            [0.0001, 100, 0.9800000190734863, 0.23697304725646973],
        ],
    ],
    "mnist_relu_6_100.onnx": [
        "MNIST",
        "6x100",
        "Standard",
        "FCN_Relu",
        6,
        [
            [0.005, 100, 0.0, 0.2672734260559082],
            [0.0005, 100, 0.1599999964237213, 0.2344954013824463],
            [0.0001, 100, 0.9700000286102295, 0.3445014953613281],
        ],
    ],
    "mnist_relu_9_100.onnx": [
        "MNIST",
        "9x100",
        "Standard",
        "FCN_Relu",
        9,
        [
            [0.005, 100, 0.0, 0.43705248832702637],
            [0.0005, 100, 0.0, 0.40650510787963867],
            [0.0001, 100, 0.6600000262260437, 0.3831460475921631],
        ],
    ],
    "mnist_relu_6_200.onnx": [
        "MNIST",
        "6x200",
        "Standard",
        "FCN_Relu",
        6,
        [
            [0.005, 100, 0.0, 0.27200984954833984],
            [0.0005, 100, 0.0, 0.26285791397094727],
            [0.0001, 100, 0.9599999785423279, 0.32694196701049805],
        ],
    ],
    "mnist_relu_9_200.onnx": [
        "MNIST",
        "9x100",
        "Standard",
        "FCN_Relu",
        9,
        [
            [0.005, 100, 0.0, 0.4790990352630615],
            [0.0005, 100, 0.0, 0.45266056060791016],
            [0.0001, 100, 0.10000000149011612, 0.4597926139831543],
        ],
    ],
    "mnist_relu_4_1024.onnx": [
        "MNIST",
        "4x1024",
        "Standard",
        "FCN_Relu",
        3,
        [
            [0.005, 100, 0.0, 0.3866603374481201],
            [0.0005, 100, 0.0, 0.35016345977783203],
            [0.0001, 100, 0.15000000596046448, 0.3566007614135742],
        ],
    ],
    "ffnnRELU__Point_6_500.onnx": [
        "MNIST",
        "6x500",
        "Standard",
        "FCN_Relu",
        6,
        [
            [0.005, 100, 0.029999999329447746, 0.48108911514282227],
            [0.0005, 100, 0.029999999329447746, 0.5160889625549316],
            [0.0001, 100, 0.029999999329447746, 0.5777866840362549],
        ],
    ],
    "ffnnTANH__Point_6_500.onnx": [
        "MNIST",
        "6x500",
        "Standard",
        "FCN_Tanh",
        6,
        [
            [0.005, 100, 0.019999999552965164, 0.31198906898498535],
            [0.0005, 100, 0.019999999552965164, 0.3340415954589844],
            [0.0001, 100, 0.019999999552965164, 0.3735620975494385],
        ],
    ],
    "convMedGTANH__Point.onnx": [
        "MNIST",
        "ConvMed",
        "Standard",
        "convolutional_Tanh",
        3,
        [
            [0.005, 100, 0.03999999910593033, 0.2238626480102539],
            [0.0005, 100, 0.03999999910593033, 0.2827305793762207],
            [0.0001, 100, 0.03999999910593033, 0.25594592094421387],
        ],
    ],
    "convSmallRELU__Point.onnx": [
        "MNIST",
        "ConvSmall",
        "Standard",
        "convolutional_Relu",
        3,
        [
            [0.005, 100, 0.10999999940395355, 0.13741064071655273],
            [0.0005, 100, 0.10999999940395355, 0.17798185348510742],
            [0.0001, 100, 0.10999999940395355, 0.1632671356201172],
        ],
    ],
    "cifar_relu_4_100.onnx": [
        "CIFAR10",
        "4x100",
        "Standard",
        "FCN_ReLU",
        4,
        [
            [4e-6, 100, 0.09000000357627869, 0.21639561653137207],
            [4e-8, 100, 0.14000000059604645, 0.21232223510742188],
        ],
    ],
    "cifar_relu_6_100.onnx": [
        "CIFAR10",
        "6x100",
        "Standard",
        "FCN_ReLU",
        6,
        [
            [4e-6, 100, 0.0, 0.3209843635559082],
            [4e-8, 100, 0.14000000059604645, 0.451282262802124],
        ],
    ],
    "ffnnRELU__PGDK_w_0.0313_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.0313",
        "FCN_ReLU",
        6,
        [
            [4e-6, 100, 0.0, 0.506103515625],
            [4e-8, 100, 0.05000000074505806, 0.5217273235321045],
        ],
    ],
    "ffnnRELU__PGDK_w_0.0078_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.0078",
        "FCN_ReLU",
        6,
        [
            [4e-6, 100, 0.029999999329447746, 0.5182621479034424],
            [4e-8, 100, 0.09000000357627869, 0.4945943355560303],
        ],
    ],
    "ffnnRELU__PGDK_w_0.1_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.1",
        "FCN_ReLU",
        6,
        [
            [4e-6, 100, 0.0, 0.42687082290649414],
            [4e-8, 100, 0.15000000596046448, 0.43461084365844727],
        ],
    ],
    "ffnnRELU__PGDK_w_0.3_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.3",
        "FCN_ReLU",
        6,
        [
            [4e-6, 100, 0.0, 0.4481642246246338],
            [4e-8, 100, 0.15000000596046448, 0.40312862396240234],
        ],
    ],
    "ffnnTANH__PGDK_w_0.3_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.0078",
        "FCN_Tanh",
        6,
        [
            [4e-6, 100, 0.0, 0.23880720138549805],
            [4e-8, 100, 0.10999999940395355, 0.24040603637695312],
        ],
    ],
    "ffnnTANH__PGDK_w_0.0313_6_500.onnx": [
        "CIFAR10",
        "6x500",
        "PGD-epsilon = 0.0313",
        "FCN_Tanh",
        6,
        [[4e-6, 100, 0.0, 0.3123018741607666], [4e-8, 100, 0.0, 0.3326852321624756]],
    ],
    "cifar_relu_7_1024.onnx": [
        "CIFAR10",
        "7x1024",
        "Standard",
        "FCN_ReLU",
        7,
        [
            [4e-6, 100, 0.0, 0.8991861343383789],
            [4e-8, 100, 0.009999999776482582, 1.1540210247039795],
        ],
    ],
    "cifar_relu_9_200.onnx": [
        "CIFAR10",
        "9x200",
        "Standard",
        "FCN_ReLU",
        9,
        [[4e-6, 100, 0.0, 0.5768587589263916], [4e-8, 100, 0.0, 0.5456869602203369]],
    ],
    "convSmallRELU__DiffAI.onnx": [
        "CIFAR10",
        "ConvSmall",
        "DiffAI",
        "convolutional_Relu",
        3,
        [
            [4e-6, 100, 0.09000000357627869, 0.1479041576385498],
            [4e-8, 100, 0.09000000357627869, 0.17593979835510254],
        ],
    ],
    "convSmallRELU__PGDK.onnx": [
        "CIFAR10",
        "ConvSmall",
        "PGD",
        "convolutional_Relu",
        3,
        [
            [4e-6, 100, 0.10999999940395355, 0.13758254051208496],
            [4e-8, 100, 0.10999999940395355, 0.13875699043273926],
        ],
    ],
    "convMedGRELU__Point.onnx": [
        "CIFAR10",
        "ConvMed",
        "Standard",
        "convolutional_Relu",
        3,
        [
            [4e-6, 100, 0.10000000149011612, 0.31096601486206055],
            [4e-8, 100, 0.10000000149011612, 0.5500783920288086],
        ],
    ],
    "convMedGRELU__PGDK_w_0.0078.onnx": [
        "CIFAR10",
        "ConvMed",
        "PGD-epsilon = 0.0078",
        "convolutional_Relu",
        3,
        [
            [4e-6, 100, 0.10000000149011612, 0.2986934185028076],
            [4e-8, 100, 0.11999999731779099, 0.3561272621154785],
        ],
    ],
    "convMedGRELU__PGDK_w_0.0313.onnx": [
        "CIFAR10",
        "ConvMed",
        "PGD-epsilon = 0.0313",
        "convolutional_Relu",
        3,
        [
            [4e-6, 100, 0.10000000149011612, 0.2864201068878176],
            [4e-8, 100, 0.11999999731779099, 0.2735321521759033],
        ],
    ],
    "convMedGTANH__PGDK_w_0.0313.onnx": [
        "CIFAR10",
        "ConvMed",
        "PGD-epsilon = 0.0313",
        "convolutional_Tanh",
        3,
        [
            [4e-6, 100, 0.0, 0.22205448150634766],
            [4e-8, 100, 0.009999999776482582, 0.21846795082092285],
        ],
    ],
    "convMedGTANH__PGDK_w_0.0078.onnx": [
        "CIFAR10",
        "ConvMed",
        "PGD-epsilon = 0.0078",
        "convolutional_Tanh",
        3,
        [
            [4e-6, 100, 0.029999999329447746, 0.2232217788696289],
            [4e-8, 100, 0.03999999910593033, 0.44502782821655273],
        ],
    ],
    "convBigRELU__DiffAI.onnx": [
        "CIFAR10",
        "ConvBig",
        "DiffAI",
        "convolutional_Relu",
        6,
        [
            [4e-6, 100, 0.05000000074505806, 0.539623498916626],
            [4e-8, 100, 0.05000000074505806, 0.6642887592315674],
        ],
    ],
    "convSuperRELU__DiffAI.onnx": [
        "CIFAR10",
        "ConvSuper",
        "DiffAI",
        "convolutional_Relu",
        6,
        [
            [4e-6, 100, 0.15000000596046448, 1.3504126071929932],
            [4e-8, 100, 0.15000000596046448, 1.2649235725402832],
        ],
    ],
}

columns = [
    "Dataset",
    "Model",
    "Training Method",
    "Architecture",
    "Layers",
    "epsilon",
    "Batch Size",
    "Precision",
    "Time",
]


rows, names = [], []
for i, (fname, attrs) in enumerate(data.items(), start=1):
    rows.append(attrs)
    names.append(f"N{i}")

df = pd.DataFrame(rows, columns=columns, index=names)
df = df.reset_index().rename(columns={"index": "Name"})


df["Precision"] = df["Precision"].round(2)
df["Time"] = df["Time"].round(2)


os.makedirs("pics", exist_ok=True)


fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
table = ax.table(
    cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.2)

plt.title("Precision-runtime comparison across different DNNs", fontsize=12, pad=12)


plt.savefig("pics/precision.png", dpi=300, bbox_inches="tight")
plt.show()
