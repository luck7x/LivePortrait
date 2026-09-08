"""Same-person reconstruction probes with true target frames, not a trainer.

Two source frames crossed with FP16/FP32 F/W/G rendering. M and final motion
coordinates stay FP32 and fixed across the precision comparison. Uses the paper
Eq.2 absolute transfer with source canonical points; no stitching, retargeting,
pasteback, face detector, temporal filter or video encoding. This is a controlled
reconstruction test, not the default relative-motion CLI or a cross-ID benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from diagnose_portrait import REPO, WorkspaceBudget, inside, sha256, validate_probe, write_json
from probe_boundary import validate_frames


def source_indices(values, count):
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError('Exactly two distinct source indices are required')
    return validate_frames(values, count)


def run(options):
    if sys.platform != 'linux':
        raise RuntimeError('Run only on the authorized Linux GPU server')
    root = options.workspace.resolve(strict=True)
    inside(REPO, root)
    video = inside(options.video, root)
    output = inside(options.output_dir, root)
    if output.exists():
        raise FileExistsError(output)
    if video.suffix.lower() != '.mp4' or video.stat().st_size > 64 * 2**20:
        raise ValueError('Expected bounded trusted MP4')
    for key in ('HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        if not os.environ.get(key):
            raise RuntimeError('Missing isolated environment: ' + key)
        inside(os.environ[key], root)
    budget = WorkspaceBudget(root, 20)
    budget.require(128 * 2**20)
    if shutil.disk_usage(root).free < 2**30:
        raise RuntimeError('Insufficient free space')
    meta = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate:format=duration',
        '-of', 'json', str(video)], text=True, timeout=40))
    count = validate_probe(meta)
    if meta['streams'][0]['width'] != meta['streams'][0]['height']:
        raise ValueError('Use a square face-centered clip; no automatic face crop in this test')
    if options.appearance_size not in (256, 512):
        raise ValueError('Only 256/512 appearance inputs are allowed')
    if options.appearance_size == 512 and meta['streams'][0]['width'] < 512:
        raise ValueError('The high-resolution arm requires at least 512 genuine input pixels')
    targets = validate_frames(options.frames, count)
    sources = source_indices(options.source_frames, count)
    selected = sorted(set(targets + sources))
    sys.path.insert(0, str(REPO))
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper

    cv2.setNumThreads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one approved GPU')
    # Keep these process-local settings fixed in both arms; FP32 is not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = InferenceConfig(flag_use_half_precision=False, device_id=0, flag_force_cpu=False,
                          flag_do_torch_compile=False)
    paths = [inside(p, root) for p in (cfg.checkpoint_F, cfg.checkpoint_M, cfg.checkpoint_W,
                                      cfg.checkpoint_G, cfg.checkpoint_S)]
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'kind': 'same_person_absolute_reconstruction_not_fix',
              'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'video_sha256': sha256(video), 'video_probe': meta, 'sources': sources, 'targets': targets,
              'weights': {str(p.relative_to(root)): sha256(p) for p in paths},
              'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'motion_precision': 'FP32 fixed',
              'render_precision': ['fp16', 'fp32'], 'tf32': False,
              'motion_input_size': 256, 'appearance_input_size': options.appearance_size,
              'expected_output_size': options.appearance_size * 2,
              'cudnn_benchmark': torch.backends.cudnn.benchmark, 'roi_256': [64, 120, 192, 240],
              'resize': 'cv2 INTER_LINEAR to 256 for all inputs; INTER_AREA output to 256 for metrics',
              'metrics': [], 'note': 'Changing source changes appearance AND source canonical geometry; not a pure texture ablation.'}
    write_json(output / 'probe.json', report)

    def save(name, image):
        budget.require(image.nbytes + 65536)
        Image.fromarray(image).save(output / (name + '.png'))

    try:
        images, images512 = {}, {}
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise RuntimeError('Video open failed')
            for i in range(max(selected) + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError('Missing decoded frame')
                if i in selected:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    images[i] = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                    save(f'{i:06d}-gt256', images[i])
                    images512[i] = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    save(f'{i:06d}-gt512', images512[i])
        finally:
            cap.release()
        wrapper = LivePortraitWrapper(cfg)
        prepared = {i: wrapper.prepare_source(images[i]) for i in selected}
        motions = {i: wrapper.get_kp_info(prepared[i]) for i in selected}
        arrays = {}
        for s in sources:
            ks = wrapper.transform_keypoint(motions[s])
            kd = {i: wrapper.transform_keypoint(dict(motions[i], kp=motions[s]['kp'])) for i in targets}
            arrays[f's{s}_kp_source'] = ks.detach().cpu().numpy()
            arrays[f's{s}_kp_target'] = np.stack([kd[i].detach().cpu().numpy() for i in targets])
            for half in (True, False):
                precision = 'fp16' if half else 'fp32'
                wrapper.inference_cfg.flag_use_half_precision = half
                wrapper.inference_cfg.input_shape = (options.appearance_size, options.appearance_size)
                appearance = images[s] if options.appearance_size == 256 else images512[s]
                feature = wrapper.extract_feature_3d(wrapper.prepare_source(appearance))
                self_rgb = wrapper.parse_output(wrapper.warp_decode(feature, ks, ks)['out'])[0]
                save(f's{s}-{precision}-self', self_rgb)
                for i in targets:
                    prediction = wrapper.warp_decode(feature, ks, kd[i])['out']
                    if not torch.isfinite(prediction).all():
                        raise RuntimeError('Non-finite prediction')
                    rgb = wrapper.parse_output(prediction)[0]
                    if rgb.shape != (options.appearance_size * 2, options.appearance_size * 2, 3):
                        raise RuntimeError('Unexpected renderer output size')
                    save(f's{s}-{precision}-f{i:06d}', rgb)
                    reduced = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
                    diff = reduced.astype(np.float32) - images[i].astype(np.float32)
                    roi = diff[120:240, 64:192]
                    mse = float(np.mean(diff**2))
                    report['metrics'].append({'source': s, 'precision': precision, 'frame': i,
                        'full_mae': float(np.mean(np.abs(diff))),
                        'full_psnr': float(10*np.log10(255**2/max(mse, 1e-12))),
                        'mouth_roi_mae': float(np.mean(np.abs(roi)))})
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError('Non-finite saved motion')
        budget.require(sum(a.nbytes for a in arrays.values()) + 65536)
        np.savez_compressed(output / 'motion.npz', **arrays)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'probe.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-frames', type=int, nargs=2, required=True)
    parser.add_argument('--appearance-size', type=int, choices=(256, 512), default=256,
                        help='512 is an out-of-training-resolution diagnostic, not a recommended setting')
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
