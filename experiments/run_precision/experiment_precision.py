import json
import os
import shutil
import sys
import time
from datetime import datetime

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constraintflow.core.compiler.compile import compile as _compile
from constraintflow.core.verifier.provesound import provesound as _provesound

mapping = {
    "N1": ["mnist_relu_3_50.onnx", "MNIST"],
    "N2": ["mnist_relu_3_100.onnx", "MNIST"],
    "N3": ["mnist_relu_5_100.onnx", "MNIST"],
    "N4": ["mnist_relu_6_100.onnx", "MNIST"],
    "N5": ["mnist_relu_9_100.onnx", "MNIST"],
    "N6": ["mnist_relu_6_200.onnx", "MNIST"],
    "N7": ["mnist_relu_9_200.onnx", "MNIST"],
    "N8": ["mnist_relu_4_1024.onnx", "MNIST"],
    "N9": ["convSmallRELU__Point.onnx", "MNIST"],
    "N10": ["convSmallRELU__DiffAI.onnx", "MNIST"],
    "N11": ["convSmallRELU__PGDK.onnx", "MNIST"],
    "N12": ["convMedGRELU__Point.onnx", "MNIST"],
    "N13": ["convMedGTANH__Point.onnx", "MNIST"],
    "N14": ["convBigRELU__DiffAI.onnx", "MNIST"],
    "N15": ["convSuperRELU__DiffAI.onnx", "MNIST"],
    "N16": ["ffnnTANH__Point_6_500.onnx", "MNIST"],
    "N17": ["ffnnTANH__PGDK_w_0.3_6_500.onnx", "MNIST"],
    "N18": ["ffnnRELU__Point_6_500.onnx", "MNIST"],
    "N19": ["ffnnRELU__PGDK_w_0.3_6_500.onnx", "MNIST"],
    "N20": ["ffnnRELU__PGDK_w_0.1_6_500.onnx", "MNIST"],
    "N21": ["cifar_relu_4_100.onnx", "CIFAR10"],
    "N22": ["cifar_relu_6_100.onnx", "CIFAR10"],
    "N23": ["cifar_relu_9_200.onnx", "CIFAR10"],
    "N24": ["cifar_relu_7_1024.onnx", "CIFAR10"],
    "N25": ["convSmallRELU__DiffAI.onnx", "CIFAR10"],
    "N26": ["convSmallRELU__PGDK.onnx", "CIFAR10"],
    "N27": ["convSmallRELU__Point.onnx", "CIFAR10"],
    "N28": ["convMedGRELU__Point.onnx", "CIFAR10"],
    "N29": ["convMedGRELU__PGDK_w_0.0078.onnx", "CIFAR10"],
    "N30": ["convMedGRELU__PGDK_w_0.0313.onnx", "CIFAR10"],
    "N31": ["convMedGTANH__Point.onnx", "CIFAR10"],
    "N32": ["convMedGTANH__PGDK_w_0.0078.onnx", "CIFAR10"],
    "N33": ["convMedGTANH__PGDK_w_0.0313.onnx", "CIFAR10"],
    "N34": ["convBigRELU__DiffAI.onnx", "CIFAR10"],
    "N35": ["ffnnRELU__Point_6_500.onnx", "CIFAR10"],
    "N36": ["ffnnRELU__PGDK_w_0.0078_6_500.onnx", "CIFAR10"],
    "N37": ["ffnnRELU__PGDK_w_0.0313_6_500.onnx", "CIFAR10"],
    "N38": ["ffnnTANH__PGDK_w_0.0313_6_500.onnx", "CIFAR10"],
    "N39": ["ffnnTANH__Point_6_500.onnx", "CIFAR10"],
    "N40": ["ffnnSIGMOID__PGDK_w_0.1_6_500.onnx", "MNIST"],
    "N41": ["ffnnSIGMOID__PGDK_w_0.3_6_500.onnx", "MNIST"],
    "N42": ["ffnnSIGMOID__Point_6_500.onnx", "MNIST"],
    "N43": ["convMedGSIGMOID__PGDK_w_0.0078.onnx", "CIFAR10"],
    "N44": ["convMedGSIGMOID__PGDK_w_0.0313.onnx", "CIFAR10"],
    "N45": ["convMedGSIGMOID__Point.onnx", "CIFAR10"],
    "N46": ["ffnnSIGMOID__PGDK_w_0.0078_6_500.onnx", "CIFAR10"],
    "N47": ["ffnnSIGMOID__PGDK_w_0.0313_6_500.onnx", "CIFAR10"],
    "N48": ["ffnnSIGMOID__Point_6_500.onnx", "CIFAR10"],
}


def get_program(program_file: str) -> str:
    return program_file


def get_network(network: str, network_format: str, dataset: str) -> str:
    if dataset not in ["mnist", "cifar"]:
        return network
    return network


def get_dataset(batch_size: int, dataset: str, train: bool = False):
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(
            root=".", train=train, download=True, transform=transform
        )
    elif dataset == "tinyimagenet":
        train = True
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # TinyImageNet images are 64x64
                transforms.ToTensor(),
            ]
        )
        root_dir = "tinyimagenet/tiny-imagenet-200"
        split = "train" if train else "test"
        data_dir = os.path.join(root_dir, split)
        if train:
            data = datasets.ImageFolder(root=data_dir, transform=transform)
        else:
            # TinyImageNet test: all images in one folder
            from torchvision.datasets.folder import default_loader

            class TinyImageNetTest(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.root = root
                    self.transform = transform
                    self.loader = default_loader
                    self.images = sorted(os.listdir(root))

                def __len__(self):
                    return len(self.images)

                def __getitem__(self, idx):
                    path = os.path.join(self.root, self.images[idx])
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, -1

            data = TinyImageNetTest(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    image, label = next(iter(dataloader))
    if dataset == "tinyimagenet":
        image = image[:, :, :56, :56]  # ensure 3 channels
    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    return image, label


def get_precision(lb):
    # print(lb, lb.shape)

    verified = (lb >= 0).all(dim=1)

    # print(verified)
    precision = verified.sum() / verified.shape[0]
    # print(precision)

    return precision


def provesound(program_file: str, nprev: int = 1, nsymb: int = 1):
    """
    Prove soundness of a ConstraintFlow program.
    """
    program = get_program(program_file)
    res = _provesound(program, nprev=nprev, nsymb=nsymb)
    print(f"Provesound result: {res}")


def compile_code(program_file: str, output_path: str = "output/"):
    """
    Compile a ConstraintFlow program into Python.
    """
    try:
        # os.makedirs(output_path, exist_ok=True)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder '{output_path}': {e}")
        sys.exit(1)

    program = get_program(program_file)
    res = _compile(program, output_path)
    if res:
        print("Compilation successful ✅")
    else:
        print("Compilation failed ❌")
        sys.exit(1)


def run(
    program_file: str,
    network: str = "mnist_relu_3_50",
    network_format: str = "onnx",
    dataset: str = "mnist",
    batch_size: int = 1,
    eps: float = 0.01,
    train: bool = False,
    print_intermediate_results: bool = False,
    no_sparsity: bool = False,
    output_path: str = "output/",
    do_compile: bool = False,
):
    """
    Run a compiled ConstraintFlow program.
    """
    if do_compile:
        """
        try:
            # os.makedirs(output_path, exist_ok=True)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating folder '{output_path}': {e}")
            sys.exit(1)
        """
        compile_code(program_file, output_path)

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)

    start_time = time.time()
    lb, ub = run(
        network_file,
        batch_size,
        eps,
        X,
        y,
        dataset=dataset,
        train=train,
        print_intermediate_results=print_intermediate_results,
        no_sparsity=no_sparsity,
    )

    duration = time.time() - start_time
    # print(f"Lower bound: {lb}")
    # print(f"Upper bound: {ub}")
    precision = get_precision(lb)

    print(f"Precision: {precision}")

    return precision, duration


def run_all():
    file = "gpt-5_deeppoly_relu.cf"
    eps_mnist = 0.0001
    eps_cifar = 4e-6
    batch_size = 100

    compile_code(program_file=file)

    for directory, dataset, eps in [
        ("nets/mnist", "mnist", eps_mnist),
        ("nets/cifar", "cifar", eps_cifar),
    ]:
        for fname in os.listdir(directory):
            net_path = os.path.join(directory, fname)
            if not fname.endswith(".onnx"):
                continue

            print(f"Running: {fname}")
            try:
                precision, duration = run(
                    program_file=file,
                    network=net_path,
                    network_format="onnx",
                    dataset=dataset,
                    batch_size=batch_size,
                    eps=eps,
                    train=False,
                    print_intermediate_results=False,
                    no_sparsity=False,
                    output_path="output/",
                    do_compile=False,
                )

                for N, vals in mapping.items():
                    mapped_fname, mapped_ds = vals[0], vals[1]

                    if fname == mapped_fname and dataset.lower() in mapped_ds.lower():
                        mapping[N].extend(
                            [eps, batch_size, float(precision), float(duration)]
                        )
                        break
                else:
                    print(f"{fname} not found in mapping")

            except Exception as e:
                print(f"Skipped {fname} due to error: {e}")

    # save to file
    rows = []
    for N, vals in mapping.items():
        fname = vals[0] if len(vals) > 0 else None
        dataset = vals[1] if len(vals) > 1 else None
        eps = vals[2] if len(vals) > 2 else None
        batch_size = vals[3] if len(vals) > 3 else None
        precision = vals[4] if len(vals) > 4 else None
        duration = vals[5] if len(vals) > 5 else None

        rows.append([N, fname, dataset, eps, batch_size, precision, duration])

    df = pd.DataFrame(
        rows,
        columns=[
            "N#",
            "fname",
            "dataset",
            "eps",
            "batch_size",
            "precision",
            "duration",
        ],
    )
    os.makedirs("precision_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = os.path.join("precision_results", f"{timestamp}.csv")
    df.to_csv(csv_file, index=False)

    print(f"Saved results to {csv_file}")


if __name__ == "__main__":
    # compile_code(program_file = "gpt-5_deeppoly_relu.cf")
    # compile_code(program_file = "test.cf")

    # run_all()

    precision, duration = run(
        program_file="constraintflow_ibp.cf",
        network="nets/cifar/convMedGTANH__Point.onnx",
        network_format="onnx",
        dataset="cifar",
        batch_size=100,
        eps=4e-6,
        train=False,
        print_intermediate_results=False,
        no_sparsity=False,
        output_path="output/",
        do_compile=True,
    )

    print(f"Duration: {duration}")
