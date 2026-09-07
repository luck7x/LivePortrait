# Human portrait diagnostics

`diagnose_portrait.py` records the existing human image-to-video pipeline without
changing model weights, motion formulas or decoder output values. It is not a
trainer, a teeth detector, or a quality score.

## Execution

Run on an approved Linux GPU server in a dedicated workspace. The original
models must already be installed. Expose exactly one approved GPU and configure
HOME, temporary directories, Hugging Face/Torch caches and XDG caches inside the
workspace. CPU affinity, thread limits and job timeouts belong to the launcher.
Do not run model inference on the development workstation.

Example (replace placeholders; these are not literal paths):

```text
python scripts/diagnose_portrait.py --workspace <workspace> --source <workspace>/data/source.jpg --driving <workspace>/data/driving.mp4 --output-dir <workspace>/runs/new-case
```

- Inputs and repo must resolve inside the workspace; use a new output directory.
- Source: JPEG/PNG, at most 32 million pixels. Driving: MP4, at most 16 seconds,
  400 actually decoded frames, 60 FPS and 1920 pixels per dimension.
- The script copies only the two case inputs, because the upstream pipeline
  writes a motion template beside the driving file.
- Defaults match `ArgumentConfig`. `--driving-multiplier` permits a single-factor
  sensitivity experiment; reduced motion is not by itself a quality improvement.
- Workspace budget cannot be raised above 20 GiB by a CLI argument. The script
  checks usage at stages and during capture, reserves metadata headroom, and
  applies a temporary per-file size limit to video/audio encoder subprocesses.
  It assumes one writer in the workspace; it is not a filesystem quota or a
  replacement for the laboratory resource policy.

## Artifacts

- `raw_frames/000000.png`: original `parse_output` **uint8 RGB before H.264**.
  PNG is lossless relative to that array, not to the floating-point decoder output.
- `model_source_256.png`: the cropped RGB image passed to the appearance encoder.
- `motion_trace.npz`: source keypoints, per-frame final driving keypoints, and
  per-frame occlusion maps. These are latent coordinates/maps, not tooth labels
  or ground-truth image optical flow.
- `videos/`: upstream output videos; frame counts are checked against PNG count.
- `diagnostics.json`: commit, input hashes, effective arguments, providers,
  elapsed time and completion/failure state.

Capturing frames/traces adds transfers and I/O, so elapsed time is not a clean
inference speed benchmark. Compare pixels and temporal behavior, not only speed.
Inspect original driver motion before labeling a normal mouth closure as missing
teeth. Inspect pre-encoding PNGs to separate model artifacts from compression.
Keep media, private paths, machine records and model weights out of Git.

CPU-only recorder/path tests (no torch/model import):

```text
python -m unittest discover -s tests -v
```
