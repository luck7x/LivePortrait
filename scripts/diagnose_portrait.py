"""Record an unmodified human i2v pipeline's pre-video RGB frames and motion.

Run only in an isolated Linux GPU workspace. This is not a trainer or a teeth
quality metric. PNGs preserve the original parse_output uint8 RGB values, not
the decoder's floating-point tensors. No media or machine paths belong in Git.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def inside(path, workspace):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(workspace).resolve()):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_budget(raw_mib, workspace_gib):
    if raw_mib <= 0 or not math.isfinite(workspace_gib) or not 0 < workspace_gib <= 20:
        raise ValueError("Positive budgets are required; workspace budget cannot exceed 20 GiB")


def validate_probe(probe):
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0
    # Actual decoded count, not duration * FPS: VFR/header estimates can be wrong.
    frames = int(stream["nb_read_frames"])
    if not math.isfinite(duration) or not 0 < duration <= 16 or not 0 < fps <= 60:
        raise ValueError("Pilot driving clip must be <=16 seconds with a valid <=60 FPS rate")
    if not 0 < frames <= 400 or max(stream["width"], stream["height"]) > 1920:
        raise ValueError("Pilot driving clip exceeds 400 actual frames or 1920-pixel input limit")
    return frames


class WorkspaceBudget:
    """Single-writer pilot guard; not an administrator-enforced filesystem quota."""
    def __init__(self, workspace, gib):
        self.workspace = Path(workspace)
        self.limit = int(gib * 2**30)

    def require(self, byte_count):
        used = int(subprocess.check_output(["du", "-s", "-B1", str(self.workspace)]).split()[0])
        # Keep headroom for metadata, small templates, log writes and block rounding.
        remaining = self.limit - used - 32 * 2**20
        if byte_count > remaining:
            raise RuntimeError("Workspace disk budget reached")
        return remaining

    def media(self, original):
        def limited(*args, **kwargs):
            import resource
            remaining = self.require(1)
            old = resource.getrlimit(resource.RLIMIT_FSIZE)
            soft = remaining if old[0] == resource.RLIM_INFINITY else min(old[0], remaining)
            resource.setrlimit(resource.RLIMIT_FSIZE, (soft, old[1]))
            try:
                return original(*args, **kwargs)
            finally:
                resource.setrlimit(resource.RLIMIT_FSIZE, old)
                self.require(0)
        return limited


class FrameCapture:
    """Wrap parse_output without changing its returned array or pixel values."""

    def __init__(self, directory, byte_limit, max_frames=400, budget_check=None):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.byte_limit = byte_limit
        self.max_frames = max_frames
        self.budget_check = budget_check
        self.bytes_written = 0
        self.count = 0

    def wrap(self, original):
        def record(tensor):
            from PIL import Image

            if self.count >= self.max_frames:
                raise RuntimeError("Pilot frame limit reached")
            frames = original(tensor)
            if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
                raise ValueError("Expected a single BxHxWx3 RGB output")
            if str(frames.dtype) != "uint8":
                raise ValueError("Expected original parse_output uint8 values")
            # Conservative upper bound for one RGB PNG plus its container overhead.
            if self.bytes_written + frames[0].nbytes + 65536 > self.byte_limit:
                raise RuntimeError("Raw-frame disk budget reached")
            if self.budget_check is not None and self.count % 16 == 0:
                self.budget_check(16 * (frames[0].nbytes + 65536))
            path = self.directory / f"{self.count:06d}.png"
            if path.exists():
                raise FileExistsError(path)
            Image.fromarray(frames[0]).save(path, compress_level=1)
            self.bytes_written += path.stat().st_size
            self.count += 1
            return frames
        return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--driving", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--driving-multiplier", type=float, default=1.0)
    p.add_argument("--raw-budget-mib", type=int, default=384)
    p.add_argument("--workspace-budget-gib", type=float, default=20.0)
    return p


def run(options):
    if sys.platform != "linux":
        raise RuntimeError("GPU diagnosis must run on the Linux server, not the local workstation")
    workspace = options.workspace.resolve(strict=True)
    inside(REPO, workspace)
    source = inside(options.source, workspace)
    driving = inside(options.driving, workspace)
    output = inside(options.output_dir, workspace)
    if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("A source photo is required")
    if not driving.is_file() or driving.suffix.lower() != ".mp4":
        raise ValueError("A driving MP4 is required; untrusted pickle input is not supported")
    if source == driving or output.exists():
        raise FileExistsError("Use distinct inputs and a new output directory")
    if not math.isfinite(options.driving_multiplier) or not 0 < options.driving_multiplier <= 2:
        raise ValueError("Driving multiplier must be finite and in (0, 2]")
    validate_budget(options.raw_budget_mib, options.workspace_budget_gib)
    for key in ("HOME", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing isolated environment variable: {key}")
        inside(os.environ[key], workspace)
    budget = WorkspaceBudget(workspace, options.workspace_budget_gib)
    raw_budget = options.raw_budget_mib * 2**20
    reserve = raw_budget + 128 * 2**20
    budget.require(reserve)
    if shutil.disk_usage(workspace).free < reserve + 2**30:
        raise RuntimeError("Insufficient filesystem free space")
    if source.stat().st_size + driving.stat().st_size > 64 * 2**20:
        raise ValueError("Pilot inputs exceed the 64 MiB per-case allowance")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames:format=duration", "-of", "json", str(driving)
    ], text=True, timeout=30))
    input_frame_count = validate_probe(probe)
    from PIL import Image
    with Image.open(source) as image:
        if image.width * image.height > 32_000_000:
            raise ValueError("Source image exceeds the pilot pixel limit")

    # Heavy imports are deliberately deferred: local tests never import the model.
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from PIL import Image
    from inference import partial_fields
    from src.config.argument_config import ArgumentConfig
    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src import live_portrait_pipeline as pipeline_module

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one approved GPU with CUDA_VISIBLE_DEVICES")
    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    for path in (inference_cfg.checkpoint_F, inference_cfg.checkpoint_M,
                 inference_cfg.checkpoint_W, inference_cfg.checkpoint_G,
                 inference_cfg.checkpoint_S, crop_cfg.landmark_ckpt_path,
                 crop_cfg.insightface_root):
        inside(path, workspace)

    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    # The official pipeline writes a motion template beside its driving input.
    # Copy only this case's two inputs, never write into the source data or repo.
    source_copy = output / "inputs" / ("source" + source.suffix.lower())
    driving_copy = output / "inputs" / "driving.mp4"
    budget.require(source.stat().st_size + driving.stat().st_size)
    shutil.copy2(source, source_copy)
    shutil.copy2(driving, driving_copy)
    args = ArgumentConfig(source=str(source_copy), driving=str(driving_copy),
                          output_dir=str(output / "videos"),
                          driving_multiplier=options.driving_multiplier)
    inference_cfg = partial_fields(InferenceConfig, dataclasses.asdict(args))
    crop_cfg = partial_fields(CropConfig, dataclasses.asdict(args))
    info = {
        "status": "running", "kind": "baseline_diagnostics_not_training",
        "commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": sha256(source), "driving_sha256": sha256(driving),
        "arguments": dataclasses.asdict(args), "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frame_format": "pre-H264 original parse_output uint8 RGB PNG",
        "input_frame_count": input_frame_count,
    }
    write_json(output / "diagnostics.json", info)
    start = time.monotonic()
    capture = None
    try:
        pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        sessions = {name: model.session for name, model in pipeline.cropper.face_analysis_wrapper.models.items()}
        sessions["landmark_203"] = pipeline.cropper.human_landmark_runner.session
        info["onnx_providers"] = {name: session.get_providers() for name, session in sessions.items()}
        if any(providers[0] != "CUDAExecutionProvider" for providers in info["onnx_providers"].values()):
            raise RuntimeError("An ONNX model fell back from CUDA")
        wrapper = pipeline.live_portrait_wrapper
        capture = FrameCapture(output / "raw_frames", raw_budget,
                               max_frames=input_frame_count, budget_check=budget.require)
        original_parse = wrapper.parse_output
        original_warp = wrapper.warp_decode
        original_prepare = wrapper.prepare_source
        points, occlusions = [], []
        source_points = None

        def record_warp(feature_3d, kp_source, kp_driving):
            nonlocal source_points
            result = original_warp(feature_3d, kp_source, kp_driving)
            if source_points is None:
                source_points = kp_source.detach().float().cpu().numpy().copy()
            points.append(kp_driving.detach().float().cpu().numpy().copy())
            if result.get("occlusion_map") is not None:
                occlusions.append(result["occlusion_map"].detach().float().cpu().numpy().copy())
            return result

        def record_prepare(image):
            path = output / "model_source_256.png"
            if not path.exists():
                budget.require(image.nbytes + 65536)
                Image.fromarray(image).save(path)
            return original_prepare(image)

        wrapper.parse_output = capture.wrap(original_parse)
        wrapper.warp_decode = record_warp
        wrapper.prepare_source = record_prepare
        original_video = pipeline_module.images2video
        original_audio = pipeline_module.add_audio_to_video
        pipeline_module.images2video = budget.media(original_video)
        pipeline_module.add_audio_to_video = budget.media(original_audio)
        try:
            video, concat = pipeline.execute(args)
        finally:
            wrapper.parse_output = original_parse
            wrapper.warp_decode = original_warp
            wrapper.prepare_source = original_prepare
            pipeline_module.images2video = original_video
            pipeline_module.add_audio_to_video = original_audio
        if not points or capture.count != len(points):
            raise RuntimeError("Frame and keypoint trace counts differ")
        arrays = {"kp_source": source_points, "kp_driving": np.stack(points)}
        if occlusions:
            if len(occlusions) != len(points):
                raise RuntimeError("Incomplete occlusion trace")
            arrays["occlusion"] = np.stack(occlusions)
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Non-finite model trace")
        budget.require(sum(value.nbytes for value in arrays.values()) + 2**20)
        np.savez_compressed(output / "motion_trace.npz", **arrays)
        videos = []
        for path in (video, concat):
            meta = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate:format=duration",
                "-of", "json", str(path)
            ], text=True, timeout=60))
            if int(meta["streams"][0]["nb_read_frames"]) != capture.count:
                raise RuntimeError("Encoded video and captured frame counts differ")
            videos.append({"file": str(Path(path).relative_to(output)), "sha256": sha256(path), "probe": meta})
        budget.require(0)
        info.update(status="completed", frames=capture.count, videos=videos,
                    raw_frame_bytes=capture.bytes_written)
    except Exception as error:
        info.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        info["elapsed_seconds"] = round(time.monotonic() - start, 3)
        info["captured_frames"] = capture.count if capture else 0
        write_json(output / "diagnostics.json", info)
    return info


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
