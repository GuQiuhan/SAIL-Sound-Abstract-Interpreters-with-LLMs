import os
import shutil
import sys

import torch
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constraintflow.core.compiler.compile import compile as _compile
from constraintflow.core.verifier.provesound import provesound as _provesound


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
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
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
    try:
        # os.makedirs(output_path, exist_ok=True)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder '{output_path}': {e}")
        sys.exit(1)

    if do_compile:
        compile_code(program_file, output_path)

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)

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

    print(f"Lower bound: {lb}")
    print(f"Upper bound: {ub}")
    precision = get_precision(lb)
    print(f"Precision: {precision}")


if __name__ == "__main__":
    # compile_code(program_file = "gpt-5_deeppoly_relu.cf")

    run(
        program_file="gpt-5_deeppoly_relu.cf",
        network="nets/mnist/mnist_relu_3_50.onnx",
        network_format="onnx",
        dataset="mnist",
        batch_size=1,
        eps=0.01,
        train=False,
        print_intermediate_results=False,
        no_sparsity=False,
        output_path="output/",
        do_compile=True,
    )
