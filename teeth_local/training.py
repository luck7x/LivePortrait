"""Opt-in single-GPU trainer. Importing this module does not import Torch."""

import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .dataset import load_train_val, sha256
from .protection import compose_allowed_region
from .objectives import (prepare_residual, color_matched_target, gradient_error,
                         mean_color_shift)

MAX_WORKSPACE_BYTES = 20 * 1024**3
SEED = 20260909


def workspace_bytes(workspace):
    # Match the project disk budget (hardlinks counted once), not logical sizes.
    return int(subprocess.check_output(['du', '-s', '-B1', str(workspace)], text=True).split()[0])


def runtime_guard(workspace, output, steps, seconds, authorized):
    if platform.system() != "Linux" or authorized is not True:
        raise ValueError("requires Linux and explicit --authorize-workspace")
    root = Path(workspace).resolve(strict=True)
    repository = Path(__file__).resolve().parents[1]
    if not root.is_dir() or not repository.is_relative_to(root):
        raise ValueError("workspace must contain this checked-out repository")
    for key in ('HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        value = os.environ.get(key)
        if not value or not Path(value).resolve().is_relative_to(root):
            raise ValueError(f'{key} must be isolated inside workspace')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip():
        raise ValueError('refuse training from a dirty code worktree')
    destination = Path(output).absolute()
    parent = destination.parent.resolve(strict=True)
    if not parent.is_relative_to(root) or destination.exists() or destination.is_symlink():
        raise ValueError("output must be a new directory inside workspace")
    if not 1 <= steps <= 1000 or not 0 < seconds <= 1800:
        raise ValueError("steps must be 1..1000 and seconds in (0,1800]")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
        raise ValueError("explicitly expose exactly one authorized GPU")
    # Reserve enough space for both bounded prediction arrays and one small model.
    if workspace_bytes(root) + 128 * 1024**2 > MAX_WORKSPACE_BYTES:
        raise ValueError("workspace plus output reserve exceeds 20 GiB")
    return root, parent / destination.name


def compose_tensor(torch, base, residual, allowed, alpha):
    candidate = (base + residual).clamp(0, 1)
    active = allowed & (alpha > 0)
    return torch.where(active, base * (1 - alpha) + candidate * alpha, base)


def weighted_loss(torch, prediction, target, residual, allowed, alpha):
    weight = torch.where(allowed, alpha, torch.zeros_like(alpha))
    denominator = weight.sum() * 3
    l1 = ((prediction - target).abs() * weight).sum() / denominator
    regularizer = (residual.square() * weight).sum() / denominator
    return l1 + 1e-4 * regularizer


def _tensors(torch, arrays):
    def rgb(key):
        return torch.from_numpy(arrays[key]).to("cuda", dtype=torch.float32).permute(0, 3, 1, 2)[None] / 255
    base = rgb("base_rgb")
    return {"base": base, "target": rgb("target_rgb"),
            "reference": base[:, 0].clone(),
            "allowed": torch.from_numpy(arrays["allowed"]).to("cuda")[None, :, None],
            "alpha": torch.from_numpy(arrays["alpha"]).to("cuda")[None, :, None]}


def _model(torch, variant):
    if variant == "B":
        from .model_b import LocalResidualB
        return LocalResidualB().to("cuda")
    if variant == "C":
        from .model_c import TemporalResidualC
        return TemporalResidualC().to("cuda")
    raise ValueError("variant must be B or C")


def _evaluate(torch, model, dataset, check_budget, objective="pixel"):
    arrays = dataset["arrays"]
    tensors = _tensors(torch, arrays)
    model.eval()
    check_budget()
    with torch.no_grad():
        residual = prepare_residual(
            torch, model(tensors["base"], tensors["reference"], tensors["allowed"]),
            tensors["allowed"], tensors["alpha"], objective)
        prediction = compose_tensor(torch, tensors["base"], residual, tensors["allowed"], tensors["alpha"])
        if not torch.isfinite(residual).all() or not torch.isfinite(prediction).all():
            raise RuntimeError('nonfinite evaluation output')
        inactive = ~(tensors["allowed"] & (tensors["alpha"] > 0)).expand_as(prediction)
        if not torch.equal(prediction[inactive], tensors["base"][inactive]):
            raise RuntimeError("float prediction changed inactive pixels")
        candidate = ((tensors["base"] + residual).clamp(0, 1)[0].permute(0, 2, 3, 1)
                     .mul(255).round().to(torch.uint8).cpu().numpy())
    check_budget()
    outputs, per_frame = [], []
    for index, frame_id in enumerate(arrays["frame_ids"]):
        output, stats = compose_allowed_region(
            arrays["base_rgb"][index], candidate[index], arrays["allowed"][index],
            arrays["alpha"][index], arrays["protected"][index])
        if not np.array_equal(output[arrays["alpha"][index] == 0],
                              arrays["base_rgb"][index][arrays["alpha"][index] == 0]):
            raise RuntimeError("uint8 prediction changed alpha-zero pixels")
        outputs.append(output)
        per_frame.append({"frame_id": int(frame_id), "outside_max_diff": stats.outside_max_diff,
                          "changed_pixel_count": stats.changed_pixel_count})
    outputs = np.stack(outputs)
    weight = arrays["alpha"][..., None].astype(np.float64)
    target = arrays["target_rgb"].astype(np.float64)
    denominator = float(weight.sum() * 3 * 255)
    def error(rgb):
        return float((np.abs(rgb.astype(np.float64) - target) * weight).sum() / denominator)
    # Metrics use the actual quantized/composited output and unchanged GT.
    with torch.no_grad():
        final = torch.from_numpy(outputs).to(device=tensors["base"].device, dtype=torch.float32)
        final = final.permute(0, 3, 1, 2)[None] / 255
        extra = {}
        for name, rgb in (("a", tensors["base"]), ("after", final)):
            extra[f"{name}_raw_gt_gradient_error"] = float(gradient_error(
                torch, rgb, tensors["target"], tensors["allowed"], tensors["alpha"]).cpu())
            extra[f"{name}_mean_rgb_shift_from_a"] = float(mean_color_shift(
                torch, rgb, tensors["base"], tensors["allowed"], tensors["alpha"]).cpu())
    return outputs, {"dataset_id": dataset["dataset_id"], "hashes": dataset["hashes"], **extra,
                     "a_weighted_l1": error(arrays["base_rgb"]), "after_weighted_l1": error(outputs),
                     "changed_pixel_count": sum(frame["changed_pixel_count"] for frame in per_frame),
                     "outside_max_diff": max(frame["outside_max_diff"] for frame in per_frame),
                     "frames": per_frame}


def _synthetic_arrays():
    random = np.random.default_rng(SEED)
    base = random.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
    allowed = np.zeros((3, 16, 16), dtype=bool)
    allowed[:, 5:11, 5:11] = True
    allowed[1] = False  # exercise a fully hidden frame for temporal state handling
    alpha = allowed.astype(np.float32) * 0.5
    alpha[:, 5, 5] = 0
    target = base.copy()
    target[allowed] = 180
    return {"base_rgb": base, "target_rgb": target, "allowed": allowed,
            "protected": ~allowed, "alpha": alpha, "frame_ids": np.arange(3)}


def run(*, workspace, output, train_record=None, val_record=None, variant="B",
        steps=100, seconds=1800, authorized=False, synthetic_smoke=False, objective="pixel"):
    if objective not in ("pixel", "structure"):
        raise ValueError("objective must be pixel or structure")
    if synthetic_smoke:
        if train_record is not None or val_record is not None:
            raise ValueError("synthetic smoke cannot access real data records")
        steps = 2
    elif train_record is None or val_record is None:
        raise ValueError("real training requires independent admitted train and val records")
    root, destination = runtime_guard(workspace, output, steps, seconds, authorized)
    started = time.monotonic()
    if synthetic_smoke:
        train = {"arrays": _synthetic_arrays(), "dataset_id": "synthetic-only", "hashes": {}}
        val = None
    else:
        train, val = load_train_val(train_record, val_record, root)
    # Process-local only: avoid the CUDA driver's default cache outside workspace.
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    repository = Path(__file__).resolve().parents[1]
    initial_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    code_paths = [Path(__file__), Path(__file__).with_name("dataset.py"),
                  Path(__file__).with_name(f"model_{variant.lower()}.py"),
                  Path(__file__).with_name("data_contract.py"), Path(__file__).with_name("protection.py"),
                  Path(__file__).with_name("objectives.py"),
                  repository / "scripts" / "train_teeth.py"]
    initial_hashes = {str(p.relative_to(repository)): sha256(p) for p in code_paths}
    # No Torch import, CUDA initialization or model construction before admission.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA device is required")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    last_disk_check = 0.0
    def check_budget(force_disk=False):
        nonlocal last_disk_check
        torch.cuda.synchronize()
        now = time.monotonic()
        if now - started >= seconds:
            raise RuntimeError("time budget exhausted; no success is recorded")
        if force_disk or now - last_disk_check >= 30:
            if workspace_bytes(root) > MAX_WORKSPACE_BYTES:
                raise RuntimeError("workspace exceeds 20 GiB")
            last_disk_check = now

    check_budget()
    model = _model(torch, variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    tensors = _tensors(torch, train["arrays"])
    losses = []
    nonzero_gradient_steps = 0
    for _ in range(steps):
        check_budget()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = prepare_residual(
            torch, model(tensors["base"], tensors["reference"], tensors["allowed"]),
            tensors["allowed"], tensors["alpha"], objective)
        prediction = compose_tensor(torch, tensors["base"], residual, tensors["allowed"], tensors["alpha"])
        target = tensors["target"]
        if objective == "structure":
            target = color_matched_target(torch, target, tensors["base"],
                                          tensors["allowed"], tensors["alpha"])
        loss = weighted_loss(torch, prediction, target, residual, tensors["allowed"], tensors["alpha"])
        if objective == "structure":
            loss = loss + gradient_error(torch, prediction, tensors["target"],
                                         tensors["allowed"], tensors["alpha"])
            loss = loss + 0.5 * mean_color_shift(torch, prediction, tensors["base"],
                                                tensors["allowed"], tensors["alpha"])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite gradient")
        if any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0 for p in model.parameters()):
            nonzero_gradient_steps += 1
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        check_budget()
    if nonzero_gradient_steps == 0:
        raise RuntimeError('no nonzero gradients; not a successful training check')
    del tensors, prediction, residual, loss, target
    train_rgb, train_metrics = _evaluate(torch, model, train, check_budget, objective)
    metrics = {"status": "not_real_training" if synthetic_smoke else "training_run_completed_not_visual_approval",
               "variant": variant, "objective": objective, "seed": SEED, "lr": 1e-4, "steps": steps,
               "losses": losses, "nonzero_gradient_steps": nonzero_gradient_steps,
               "train": train_metrics,
               "split_claim": "dataset_id only; not identity-level",
               "quality_claim": "pixel error is not evidence of teeth improvement",
               "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0)}
    predictions = {"train_rgb": train_rgb, "train_frame_ids": train["arrays"]["frame_ids"]}
    if val is not None:
        val_rgb, metrics["val"] = _evaluate(torch, model, val, check_budget, objective)
        predictions.update(val_rgb=val_rgb, val_frame_ids=val["arrays"]["frame_ids"])
    repository = Path(__file__).resolve().parents[1]
    metrics["code_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    metrics["git_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    if metrics['git_dirty'] or metrics['code_sha'] != initial_sha:
        raise RuntimeError('code changed during the job')
    metrics["code_hashes"] = {str(p.relative_to(repository)): sha256(p) for p in code_paths}
    if metrics["code_hashes"] != initial_hashes:
        raise RuntimeError('code hashes changed during the job')
    check_budget(force_disk=True)
    destination.mkdir(exist_ok=False)
    # Smoke writes only its report: never a checkpoint or real-data predictions.
    if not synthetic_smoke:
        torch.save({"variant": variant, "objective": objective,
                    "state_dict": model.state_dict(), "steps": steps,
                    "seed": SEED, "code_sha": metrics["code_sha"],
                    "code_hashes": metrics["code_hashes"],
                    "data_hashes": {"train": train["hashes"], "val": val["hashes"]}},
                   destination / "checkpoint.pt")
        np.savez_compressed(destination / "predictions.npz", **predictions)
        metrics["checkpoint_sha256"] = sha256(destination / "checkpoint.pt")
        metrics["predictions_sha256"] = sha256(destination / "predictions.npz")
    check_budget()
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    metrics["workspace_bytes_before_report"] = workspace_bytes(root)
    with (destination / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return metrics
