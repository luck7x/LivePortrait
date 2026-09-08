"""Bounded replay probes for existing trusted diagnosis artifacts; not a fix/trainer.

Use the same isolated Linux GPU workspace as diagnose_portrait.py. Replays saved
final keypoints, so no motion extraction, stitching, pasteback or video encoding
is re-executed. Counterfactuals can be out of distribution: never rank them as
production improvements merely because they expose fewer teeth.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from diagnose_portrait import REPO, WorkspaceBudget, inside, sha256, write_json


def validate_frames(values, count):
    frames = sorted(set(values))
    if not frames or len(frames) > 12 or min(frames) < 0 or max(frames) >= count:
        raise ValueError('Select 1 to 12 existing frame indices')
    return frames


def run(options):
    if sys.platform != 'linux':
        raise RuntimeError('Model probes must run on the authorized Linux GPU server')
    root = options.workspace.resolve(strict=True)
    inside(REPO, root)
    case = inside(options.case, root)
    output = inside(options.output_dir, root)
    if output.exists():
        raise FileExistsError(output)
    for key in ('HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        if not os.environ.get(key):
            raise RuntimeError('Missing isolated environment: ' + key)
        inside(os.environ[key], root)
    budget = WorkspaceBudget(root, 20)
    budget.require(128 * 2**20)
    if shutil.disk_usage(root).free < 2**30:
        raise RuntimeError('Insufficient free space')
    trace = inside(case / 'motion_trace.npz', root)
    source = inside(case / 'model_source_256.png', root)
    metadata = inside(case / 'diagnostics.json', root)
    info = json.loads(metadata.read_text())
    if info['status'] != 'completed':
        raise ValueError('Only completed trusted diagnosis cases are accepted')
    frames = validate_frames(options.frames, info['frames'])
    with zipfile.ZipFile(trace) as archive:
        if sum(x.file_size for x in archive.infolist()) > 32 * 2**20:
            raise ValueError('Trace exceeds uncompressed size budget')
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from PIL import Image
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one approved GPU')
    image = Image.open(source).convert('RGB')
    if image.size != (256, 256):
        raise ValueError('Expected recorded 256-square source input')
    with np.load(trace, allow_pickle=False) as data:
        xs, xd = data['kp_source'], data['kp_driving']
    if xs.shape != (1, 21, 3) or xd.shape != (info['frames'], 1, 21, 3):
        raise ValueError('Unexpected saved keypoint shapes')
    if not np.isfinite(xs).all() or not np.isfinite(xd).all():
        raise ValueError('Non-finite keypoints')
    allowed = {f.name for f in dataclasses.fields(InferenceConfig)}
    settings = {k: v for k, v in info['arguments'].items() if k in allowed}
    settings.update(flag_do_torch_compile=False, device_id=0, flag_force_cpu=False)
    cfg = InferenceConfig(**settings)
    checkpoints = [inside(p, root) for p in (cfg.checkpoint_F, cfg.checkpoint_M,
                                            cfg.checkpoint_W, cfg.checkpoint_G, cfg.checkpoint_S)]
    weights = {str(p.relative_to(root)): sha256(p) for p in checkpoints}
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'kind': 'counterfactual_boundary_probe_not_fix',
              'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'original_commit': info['commit'], 'case': str(case), 'frames': frames,
              'input_hashes': {'source': sha256(source), 'trace': sha256(trace)},
              'weights': weights,
              'config': {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                         if f.name not in ('mask_crop', 'lip_array')},
              'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'comparisons': []}
    write_json(output / 'probe.json', report)
    try:
        wrapper = LivePortraitWrapper(cfg)
        feature = wrapper.extract_feature_3d(wrapper.prepare_source(np.array(image)))
        ks = torch.from_numpy(xs).to(wrapper.device)
        warper = wrapper.warping_module
        captured = {}
        hook = warper.fourth.register_forward_hook(lambda module, args, result: captured.update(pre_gate=result.detach()))

        def pixels(tensor):
            return wrapper.parse_output(tensor.float())[0]

        def save(name, array):
            budget.require(array.nbytes + 65536)
            Image.fromarray(array).save(output / (name + '.png'))

        def predict(kp):
            with torch.no_grad(), wrapper.inference_ctx():
                result = warper(feature, kp_source=ks, kp_driving=kp)
                base = pixels(wrapper.spade_generator(result['out']))
                no_gate = pixels(wrapper.spade_generator(captured['pre_gate']))
            return base, no_gate, result

        try:
            self_base, self_no_gate, self_result = predict(ks)
            save('self-native', self_base)
            save('self-no-gate', self_no_gate)
            # At zero keypoint motion, bypass ONLY the final feature resampling;
            # retain the learned self occlusion map and identical projection/G.
            with torch.no_grad(), wrapper.inference_ctx():
                b, c, d, h, w = feature.shape
                direct = warper.fourth(warper.third(feature.reshape(b, c*d, h, w)))
                direct = direct * self_result['occlusion_map']
                save('self-no-final-resampling', pixels(wrapper.spade_generator(direct)))
            for i in frames:
                base, no_gate, result = predict(torch.from_numpy(xd[i]).to(wrapper.device))
                repeat = pixels(wrapper.warp_decode(feature, ks, torch.from_numpy(xd[i]).to(wrapper.device))['out'])
                save(f'{i:06d}-replay', base)
                save(f'{i:06d}-no-gate', no_gate)
                old_path = inside(case / 'raw_frames' / f'{i:06d}.png', root)
                old = np.array(Image.open(old_path).convert('RGB'))
                if old.shape != base.shape:
                    raise ValueError('Original output shape mismatch')
                diff = np.abs(base.astype(np.int16)-old.astype(np.int16))
                report['comparisons'].append({'frame': i, 'replay_vs_saved_mae': float(diff.mean()),
                    'replay_vs_saved_max': int(diff.max()), 'repeat_max': int(np.abs(base.astype(np.int16)-repeat.astype(np.int16)).max())})
                arrays = {k: result[k].detach().float().cpu().numpy() for k in ('deformation', 'occlusion_map')}
                budget.require(sum(a.nbytes for a in arrays.values()) + 65536)
                np.savez_compressed(output / f'{i:06d}-fields.npz', **arrays)
        finally:
            hook.remove()
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
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
