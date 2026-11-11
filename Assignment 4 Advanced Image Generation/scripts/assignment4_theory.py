"""Execute all numerical/autograd questions and record the actual results."""

import json
from pathlib import Path
import torch


def main():
    result = {"torch_version": torch.__version__}
    j = torch.arange(4, dtype=torch.float64)
    angles = 10000 ** (-2 * j / 8)
    result["Q2"] = torch.stack((angles.sin(), angles.cos()), 1).flatten().tolist()
    x = torch.tensor([2.0], requires_grad=True)
    (x**2 + 3 * x).backward()
    result["Q6a"] = x.grad.tolist()
    assert x.grad.item() == 7
    x = torch.tensor([2.0])
    y = x**2 + 3 * x
    try:
        y.backward()
    except RuntimeError as error:
        result["Q6b_error"] = str(error)
    result["Q6c_and_Q7c_default"] = x.requires_grad
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    (w[0] * x**2 + w[1] * x).backward()
    result["Q7a"] = {
        "x_grad": x.grad.tolist(),
        "w_grad": None if w.grad is None else w.grad.tolist(),
    }
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    (w[0] * x**2 + w[1] * x).backward()
    result["Q7b"] = {"x_grad": x.grad.tolist(), "w_grad": w.grad.tolist()}
    assert w.grad.tolist() == [4.0, 2.0]
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.detach()
    w = z * 2
    try:
        w.backward()
    except RuntimeError as error:
        result["Q8_original_error"] = str(error)
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y
    w = z * 2
    w.backward()
    result["Q8_fixed"] = x.grad.tolist()
    assert x.grad.item() == 6
    x = torch.tensor([1.0], requires_grad=True)
    (x * 2).backward()
    first = x.grad.item()
    (x * 3).backward()
    second = x.grad.item()
    x.grad = None
    (x * 3).backward()
    result["Q9"] = {"first": first, "second": second, "after_reset": x.grad.item()}
    assert (first, second, x.grad.item()) == (2.0, 5.0, 3.0)
    output = (
        Path(__file__).resolve().parents[1] / "results/assignment4/theory_results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
