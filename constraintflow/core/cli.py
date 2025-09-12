import os
import sys

import torch
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constraintflow.core.compiler.compile import compile as _compile
from constraintflow.core.verifier.provesound import provesound as _provesound

app = typer.Typer(
    help="ConstraintFlow CLI for verification and compilation of DSL programs."
)


# --------------------------
# Utility Functions
# --------------------------


def get_program(program_file: str) -> str:
    return program_file


def get_network(network: str, network_format: str, dataset: str) -> str:
    return f"nets/{dataset}/{network}.{network_format}"


def get_dataset(batch_size: int, dataset: str, train: bool = False):
    transform = transforms.ToTensor()
    if dataset == "mnist":
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    else:
        data = datasets.CIFAR10(
            root=".", train=train, download=True, transform=transform
        )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    image, _ = next(iter(dataloader))
    true_label = data.targets[:batch_size]
    if isinstance(true_label, list):
        true_label = torch.tensor(true_label)
    return image, true_label


def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision


# --------------------------
# CLI Commands
# --------------------------


@app.command()
def provesound(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    nprev: int = typer.Option(1, help="Number of previous states"),
    nsymb: int = typer.Option(1, help="Number of symbols"),
):
    """
    Prove soundness of a ConstraintFlow program.
    """
    program = get_program(program_file)
    res = _provesound(program, nprev=nprev, nsymb=nsymb)
    typer.echo(f"Provesound result: {res}")


def compile_code(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
):
    """
    Compile a ConstraintFlow program into Python.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    program = get_program(program_file)
    res = _compile(program, output_path)
    if res:
        typer.echo("Compilation successful ✅")
    else:
        typer.echo("Compilation failed ❌")
        raise typer.Exit(code=1)


@app.command()
def compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
):
    compile_code(program_file, output_path)


@app.command()
def run(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option("mnist_relu_3_50", help="Network name"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0.01, help="Epsilon"),
    train: bool = typer.Option(False, help="Run on training dataset"),
    print_intermediate_results: bool = typer.Option(
        False, help="Print intermediate results"
    ),
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    output_path: str = typer.Option(
        "output/", help="Path where compiled program is stored"
    ),
    compile: bool = typer.Option(False, help="Run compilation before execution"),
):
    """
    Run a compiled ConstraintFlow program.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    if compile:
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

    typer.echo(f"Lower bound: {lb}")
    typer.echo(f"Upper bound: {ub}")
    precision = get_precision(lb)
    typer.echo(f"Precision: {precision}")


def main():
    app()


if __name__ == "__main__":
    main()
