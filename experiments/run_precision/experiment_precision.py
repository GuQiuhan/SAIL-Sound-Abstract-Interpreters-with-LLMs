import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort
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
    "N13": ["ffnnRELU__Point_6_500.onnx", "MNIST"],
    "N14": ["ffnnRELU__PGDK_w_0.3_6_500.onnx", "MNIST"],
    "N15": ["ffnnRELU__PGDK_w_0.1_6_500.onnx", "MNIST"],
    "N16": ["convBigRELU__DiffAI.onnx", "MNIST"],
    "N17": ["convSuperRELU__DiffAI.onnx", "MNIST"],
    "N18": ["ffnnSIGMOID__PGDK_w_0.1_6_500.onnx", "MNIST"],
    "N19": ["ffnnSIGMOID__PGDK_w_0.3_6_500.onnx", "MNIST"],
    "N20": ["ffnnSIGMOID__Point_6_500.onnx", "MNIST"],
    "N21": ["mnist_convmed_relu6_diffai_eps0.3.onnx", "MNIST"],
    "N22": ["mnist_convmed_relu6_pgd_eps0.1.onnx", "MNIST"],
    "N23": ["mnist_convmed_relu6_pgd_eps0.3.onnx", "MNIST"],
    "N24": ["mnist_convmed_relu6_standard_eps0.3.onnx", "MNIST"],
    "N25": ["mnist_convsmall_relu6_diffai_eps0.3.onnx", "MNIST"],
    "N26": ["mnist_convsmall_relu6_pgd_eps0.3.onnx", "MNIST"],
    "N27": ["mnist_convsmall_relu6_standard_eps0.3.onnx", "MNIST"],
    "N28": ["mnist_fcn3x50_relu6_standard_eps0.3.onnx", "MNIST"],
    "N29": ["mnist_fcn3x100_relu6_standard_eps0.3.onnx", "MNIST"],
    "N30": ["mnist_fcn4x1024_relu6_standard_eps0.3.onnx", "MNIST"],
    "N31": ["mnist_fcn5x100_relu6_diffai_eps0.3.onnx", "MNIST"],
    "N32": ["mnist_fcn6x100_relu6_standard_eps0.3.onnx", "MNIST"],
    "N33": ["mnist_fcn6x200_relu6_standard_eps0.3.onnx", "MNIST"],
    "N34": ["mnist_fcn6x500_relu6_pgd_eps0.1.onnx", "MNIST"],
    "N35": ["mnist_fcn6x500_relu6_pgd_eps0.3.onnx", "MNIST"],
    "N36": ["mnist_fcn6x500_relu6_standard_eps0.3.onnx", "MNIST"],
    "N37": ["mnist_fcn9x100_relu6_standard_eps0.3.onnx", "MNIST"],
    "N38": ["mnist_fcn9x200_relu6_standard_eps0.3.onnx", "MNIST"],
    "N39": ["mnist_convmed_hardtanh_diffai_eps0.3.onnx", "MNIST"],
    "N40": ["mnist_convmed_hardtanh_pgd_eps0.1.onnx", "MNIST"],
    "N41": ["mnist_convmed_hardtanh_pgd_eps0.3.onnx", "MNIST"],
    "N42": ["mnist_convmed_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N43": ["mnist_convsmall_hardtanh_diffai_eps0.3.onnx", "MNIST"],
    "N44": ["mnist_convsmall_hardtanh_pgd_eps0.3.onnx", "MNIST"],
    "N45": ["mnist_convsmall_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N46": ["mnist_fcn3x50_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N47": ["mnist_fcn3x100_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N48": ["mnist_fcn4x1024_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N49": ["mnist_fcn5x100_hardtanh_diffai_eps0.3.onnx", "MNIST"],
    "N50": ["mnist_fcn6x100_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N51": ["mnist_fcn6x200_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N52": ["mnist_fcn6x500_hardtanh_pgd_eps0.1.onnx", "MNIST"],
    "N53": ["mnist_fcn6x500_hardtanh_pgd_eps0.3.onnx", "MNIST"],
    "N54": ["mnist_fcn6x500_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N55": ["mnist_fcn9x100_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N56": ["mnist_fcn9x200_hardtanh_standard_eps0.3.onnx", "MNIST"],
    "N57": ["mnist_convmed_hardswish_diffai_eps0.3.onnx", "MNIST"],
    "N58": ["mnist_convmed_hardswish_pgd_eps0.1.onnx", "MNIST"],
    "N59": ["mnist_convmed_hardswish_pgd_eps0.3.onnx", "MNIST"],
    "N60": ["mnist_convmed_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N61": ["mnist_convsmall_hardswish_diffai_eps0.3.onnx", "MNIST"],
    "N62": ["mnist_convsmall_hardswish_pgd_eps0.3.onnx", "MNIST"],
    "N63": ["mnist_convsmall_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N64": ["mnist_fcn3x50_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N65": ["mnist_fcn3x100_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N66": ["mnist_fcn4x1024_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N67": ["mnist_fcn5x100_hardswish_diffai_eps0.3.onnx", "MNIST"],
    "N68": ["mnist_fcn6x100_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N69": ["mnist_fcn6x200_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N70": ["mnist_fcn6x500_hardswish_pgd_eps0.1.onnx", "MNIST"],
    "N71": ["mnist_fcn6x500_hardswish_pgd_eps0.3.onnx", "MNIST"],
    "N72": ["mnist_fcn6x500_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N73": ["mnist_fcn9x100_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N74": ["mnist_fcn9x200_hardswish_standard_eps0.3.onnx", "MNIST"],
    "N75": ["mnist_convmed_hardsigmoid_diffai_eps0.3.onnx", "MNIST"],
    "N76": ["mnist_convmed_hardsigmoid_pgd_eps0.1.onnx", "MNIST"],
    "N77": ["mnist_convmed_hardsigmoid_pgd_eps0.3.onnx", "MNIST"],
    "N78": ["mnist_convmed_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N79": ["mnist_convsmall_hardsigmoid_diffai_eps0.3.onnx", "MNIST"],
    "N80": ["mnist_convsmall_hardsigmoid_pgd_eps0.3.onnx", "MNIST"],
    "N81": ["mnist_convsmall_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N82": ["mnist_fcn3x50_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N83": ["mnist_fcn3x100_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N84": ["mnist_fcn4x1024_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N85": ["mnist_fcn5x100_hardsigmoid_diffai_eps0.3.onnx", "MNIST"],
    "N86": ["mnist_fcn6x100_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N87": ["mnist_fcn6x200_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N88": ["mnist_fcn6x500_hardsigmoid_pgd_eps0.1.onnx", "MNIST"],
    "N89": ["mnist_fcn6x500_hardsigmoid_pgd_eps0.3.onnx", "MNIST"],
    "N90": ["mnist_fcn6x500_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N91": ["mnist_fcn9x100_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N92": ["mnist_fcn9x200_hardsigmoid_standard_eps0.3.onnx", "MNIST"],
    "N93": ["cifar_relu_4_100.onnx", "CIFAR10"],
    "N94": ["cifar_relu_6_100.onnx", "CIFAR10"],
    "N95": ["cifar_relu_9_200.onnx", "CIFAR10"],
    "N96": ["cifar_relu_7_1024.onnx", "CIFAR10"],
    "N97": ["convSmallRELU__DiffAI.onnx", "CIFAR10"],
    "N98": ["convSmallRELU__PGDK.onnx", "CIFAR10"],
    "N99": ["convSmallRELU__Point.onnx", "CIFAR10"],
    "N100": ["convMedGRELU__Point.onnx", "CIFAR10"],
    "N101": ["convMedGRELU__PGDK_w_0.0078.onnx", "CIFAR10"],
    "N102": ["convMedGRELU__PGDK_w_0.0313.onnx", "CIFAR10"],
    "N103": ["convBigRELU__DiffAI.onnx", "CIFAR10"],
    "N104": ["ffnnRELU__Point_6_500.onnx", "CIFAR10"],
    "N105": ["ffnnRELU__PGDK_w_0.0078_6_500.onnx", "CIFAR10"],
    "N106": ["ffnnRELU__PGDK_w_0.0313_6_500.onnx", "CIFAR10"],
    "N107": ["convMedGSIGMOID__PGDK_w_0.0078.onnx", "CIFAR10"],
    "N108": ["convMedGSIGMOID__PGDK_w_0.0313.onnx", "CIFAR10"],
    "N109": ["convMedGSIGMOID__Point.onnx", "CIFAR10"],
    "N110": ["ffnnSIGMOID__PGDK_w_0.0078_6_500.onnx", "CIFAR10"],
    "N111": ["ffnnSIGMOID__PGDK_w_0.0313_6_500.onnx", "CIFAR10"],
    "N112": ["ffnnSIGMOID__Point_6_500.onnx", "CIFAR10"],
    "N113": ["cifar10_convmed_relu6_diffai_eps0.0313.onnx", "CIFAR10"],
    "N114": ["cifar10_convmed_relu6_pgd_eps0.0078.onnx", "CIFAR10"],
    "N115": ["cifar10_convmed_relu6_pgd_eps0.0313.onnx", "CIFAR10"],
    "N116": ["cifar10_convmed_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N117": ["cifar10_convsmall_relu6_diffai_eps0.0313.onnx", "CIFAR10"],
    "N118": ["cifar10_convsmall_relu6_pgd_eps0.0313.onnx", "CIFAR10"],
    "N119": ["cifar10_convsmall_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N120": ["cifar10_fcn4x100_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N121": ["cifar10_fcn6x100_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N122": ["cifar10_fcn6x500_relu6_pgd_eps0.0078.onnx", "CIFAR10"],
    "N123": ["cifar10_fcn6x500_relu6_pgd_eps0.0313.onnx", "CIFAR10"],
    "N124": ["cifar10_fcn6x500_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N125": ["cifar10_fcn7x1024_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N126": ["cifar10_fcn9x200_relu6_standard_eps0.3.onnx", "CIFAR10"],
    "N127": ["cifar10_convmed_hardtanh_diffai_eps0.0313.onnx", "CIFAR10"],
    "N128": ["cifar10_convmed_hardtanh_pgd_eps0.0078.onnx", "CIFAR10"],
    "N129": ["cifar10_convmed_hardtanh_pgd_eps0.0313.onnx", "CIFAR10"],
    "N130": ["cifar10_convmed_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N131": ["cifar10_convsmall_hardtanh_diffai_eps0.0313.onnx", "CIFAR10"],
    "N132": ["cifar10_convsmall_hardtanh_pgd_eps0.0313.onnx", "CIFAR10"],
    "N133": ["cifar10_convsmall_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N134": ["cifar10_fcn4x100_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N135": ["cifar10_fcn6x100_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N136": ["cifar10_fcn6x500_hardtanh_pgd_eps0.0078.onnx", "CIFAR10"],
    "N137": ["cifar10_fcn6x500_hardtanh_pgd_eps0.0313.onnx", "CIFAR10"],
    "N138": ["cifar10_fcn6x500_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N139": ["cifar10_fcn7x1024_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N140": ["cifar10_fcn9x200_hardtanh_standard_eps0.3.onnx", "CIFAR10"],
    "N141": ["cifar10_convmed_hardswish_diffai_eps0.0313.onnx", "CIFAR10"],
    "N142": ["cifar10_convmed_hardswish_pgd_eps0.0078.onnx", "CIFAR10"],
    "N143": ["cifar10_convmed_hardswish_pgd_eps0.0313.onnx", "CIFAR10"],
    "N144": ["cifar10_convmed_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N145": ["cifar10_convsmall_hardswish_diffai_eps0.0313.onnx", "CIFAR10"],
    "N146": ["cifar10_convsmall_hardswish_pgd_eps0.0313.onnx", "CIFAR10"],
    "N147": ["cifar10_convsmall_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N148": ["cifar10_fcn4x100_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N149": ["cifar10_fcn6x100_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N150": ["cifar10_fcn6x500_hardswish_pgd_eps0.0078.onnx", "CIFAR10"],
    "N151": ["cifar10_fcn6x500_hardswish_pgd_eps0.0313.onnx", "CIFAR10"],
    "N152": ["cifar10_fcn6x500_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N153": ["cifar10_fcn7x1024_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N154": ["cifar10_fcn9x200_hardswish_standard_eps0.3.onnx", "CIFAR10"],
    "N155": ["cifar10_convmed_hardsigmoid_diffai_eps0.0313.onnx", "CIFAR10"],
    "N156": ["cifar10_convmed_hardsigmoid_pgd_eps0.0078.onnx", "CIFAR10"],
    "N157": ["cifar10_convmed_hardsigmoid_pgd_eps0.0313.onnx", "CIFAR10"],
    "N158": ["cifar10_convmed_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N159": ["cifar10_convsmall_hardsigmoid_diffai_eps0.0313.onnx", "CIFAR10"],
    "N160": ["cifar10_convsmall_hardsigmoid_pgd_eps0.0313.onnx", "CIFAR10"],
    "N161": ["cifar10_convsmall_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N162": ["cifar10_fcn4x100_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N163": ["cifar10_fcn6x100_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N164": ["cifar10_fcn6x500_hardsigmoid_pgd_eps0.0078.onnx", "CIFAR10"],
    "N165": ["cifar10_fcn6x500_hardsigmoid_pgd_eps0.0313.onnx", "CIFAR10"],
    "N166": ["cifar10_fcn6x500_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N167": ["cifar10_fcn7x1024_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
    "N168": ["cifar10_fcn9x200_hardsigmoid_standard_eps0.3.onnx", "CIFAR10"],
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


def get_precision_new(lb: torch.Tensor, baseline_correct_mask: torch.Tensor) -> float:
    certified = (lb >= 0).all(dim=1)
    denom = baseline_correct_mask.sum().item()  # baseline

    if denom == 0:

        return float("inf")

    num = (certified & baseline_correct_mask).sum().item()
    return num / denom


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


def baseline_correct_mask_onnx(
    network_file: str, X: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    x_np = X.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy()

    def onnx_forward_auto_batch(
        sess: ort.InferenceSession, x_np: np.ndarray
    ) -> np.ndarray:
        inp = sess.get_inputs()[0]
        inp_name = inp.name
        shape = inp.shape
        fixed_bs1 = False
        if len(shape) >= 1:
            dim0 = shape[0]
            if isinstance(dim0, int) and dim0 == 1:
                fixed_bs1 = True

        if fixed_bs1:
            outs = []
            for i in range(x_np.shape[0]):
                out_i = sess.run(None, {inp_name: x_np[i : i + 1]})[0]  # [1, C]
                outs.append(out_i)
            return np.concatenate(outs, axis=0)  # [B, C]
        else:
            return sess.run(None, {inp_name: x_np})[0]  # [B, C]

    sess = ort.InferenceSession(network_file, providers=["CPUExecutionProvider"])
    logits = onnx_forward_auto_batch(sess, x_np)  # [B, num_classes]
    preds = logits.argmax(axis=1)
    correct = preds == y_np
    return torch.from_numpy(correct)


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

    baseline_mask = baseline_correct_mask_onnx(network_file, X, y)

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
    # precision = get_precision(lb)
    precision = get_precision_new(lb, baseline_mask)

    print(f"Precision: {precision}")

    return precision, duration


def run_all(file):
    eps_mnist = [0.005, 0.0005]
    eps_cifar = [4e-6, 4e-8]
    batch_size = 100

    compile_code(program_file=file)

    for directory, dataset, eps_list in [
        ("nets/mnist", "mnist", eps_mnist),
        ("nets/cifar", "cifar", eps_cifar),
    ]:
        for fname in os.listdir(directory):
            for eps in eps_list:
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

                        if (
                            fname == mapped_fname
                            and dataset.lower() in mapped_ds.lower()
                        ):
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

    # compile_code(program_file = "gpt-5_deeppoly_nonlinear.cf")

    # run_all(file = "gpt-5_deeppoly.cf")
    # run_all(file = "constraintflow_deepz.cf")
    # run_all(file = "gpt-5_deepz.cf")
    # run_all(file="llama4_deeppoly.cf")

    precision, duration = run(
        program_file="gpt-5_deeppoly_nonlinear.cf",
        network="nets/mnist/mnist_fcn3x100_gelu_standard_eps0.3.onnx",
        network_format="onnx",
        dataset="mnist",
        batch_size=100,
        eps=0.005,
        train=False,
        print_intermediate_results=False,
        no_sparsity=False,
        output_path="output/",
        do_compile=True,
    )

    print(f"Duration: {duration}")
