"""E1: bounded per-frame free tail-feature capacity, NOT a deployable model/video."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.snapshot_upper_teeth import (OwnedProcessGroup, array_hash, budget,
                                         code_check, command, inside, require, sha256)

OPEN = {228, 264, 266}
CLOSED = {251, 252, 292}
CONTACT = {225}
RADII = (.05, .1, .2)
NORMALIZED_TOLERANCE = .002  # Fixed absolute allowance for the actual FP16 addition.


def check_actual_radius(actual_max, radius):
    require(math.isfinite(actual_max) and actual_max >= 0, 'numerical_failure: invalid actual residual')
    require(actual_max == 0 if radius == 0 else actual_max <= radius + NORMALIZED_TOLERANCE,
            'numerical_failure: actual normalized residual exceeds fixed radius tolerance')


def bounded_npz(path, limit_mib, keys=None):
    limit = limit_mib * 2**20
    require(Path(path).stat().st_size <= limit, 'NPZ compressed size limit')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 64 and len({x.filename for x in entries}) == len(entries)
                and sum(x.file_size for x in entries) <= limit, 'NPZ expanded size limit/duplicate entries')
        if keys is not None:
            require({x.filename for x in entries} == {k + '.npy' for k in keys}, 'NPZ keys mismatch')
        declared_bytes = 0
        for entry in entries:
            require(entry.filename.endswith('.npy'), 'NPZ must contain only NPY arrays')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'unsupported NPY header version')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(stream)
                require(not dtype.hasobject and len(shape) <= 8 and all(n >= 0 for n in shape), 'unsafe NPY dtype/shape')
                size = math.prod(shape) * dtype.itemsize
                declared_bytes += size
                require(size <= entry.file_size - stream.tell() and declared_bytes <= limit,
                        'NPZ declared array size exceeds payload/limit')
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    require(sum(a.nbytes for a in result.values()) <= limit, 'NPZ array size limit')
    return result


def read_raw_png(path):
    from PIL import Image
    require(Path(path).stat().st_size <= 16 * 2**20, 'PNG compressed size limit')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.size == (512, 512) and image.mode == 'RGB',
                'expected RGB PNG512 before pixel decode')
        return np.asarray(image).copy()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record_file(value, directory, workspace):
    require(isinstance(value, str) and not Path(value).is_absolute(), 'record paths must be relative')
    path = inside(directory / value, directory)
    inside(path, workspace)
    require(path.is_file(), 'record asset must be a file')
    return path


def validate_pixels(raw, target, masks, role):
    require(raw.shape == target.shape == (512, 512, 3) and
            raw.dtype == target.dtype == np.uint8, 'expected RGB uint8 512 PNG')
    require(set(masks) == {'allowed', 'protected', 'uncertain', 'target_teeth'}, 'mask keys mismatch')
    for value in masks.values():
        require(value.shape == (512, 512) and value.dtype == np.bool_, 'masks must be bool512')
    allowed = masks['allowed']
    require(not np.any(allowed & (masks['protected'] | masks['uncertain'])), 'unsafe allowed intersection')
    require(np.array_equal(target[~allowed], raw[~allowed]), 'target changes outside allowed')
    if role in ('closed', 'protected-contact'):
        require(not allowed.any() and np.array_equal(target, raw), role + ' negative must be exact baseline')
    else:
        require(allowed.any() and np.any(target != raw), 'open counterfactual must change allowed pixels')


def validate_record(path, snapshot, report, workspace):
    record = read_json(path)
    for key, expected in {'schema': 'upper-tail-counterfactual-v1',
                          'purpose': 'diagnostic_counterfactual_only',
                          'user_authorized_experiment': True,
                          'human_semantic_mask_approved': False,
                          'independent_review': 'passed',
                          'snapshot_code_sha': report['code_sha']}.items():
        require(type(record.get(key)) is type(expected) and record[key] == expected, 'record gate: ' + key)
    require(isinstance(record.get('frames'), list) and record['frames'], 'empty frame record')
    seen, assets = set(), {str(path): sha256(path)}
    frames = []
    for item in record['frames']:
        frame, role = item['frame'], item['role']
        require(type(frame) is int and frame not in seen and
                ((role == 'open' and frame in OPEN) or (role == 'closed' and frame in CLOSED)
                 or (role == 'protected-contact' and frame in CONTACT)),
                'invalid/duplicate frame or role; never fit 446/456')
        seen.add(frame)
        baseline = snapshot / f'raw512_f{frame:04d}.png'
        require(sha256(baseline) == item['baseline_sha256'] == report['files'][baseline.name], 'baseline hash mismatch')
        paths = {}
        for kind in ('target', 'masks'):
            p = record_file(item[kind], path.parent, workspace)
            require(p.suffix.lower() == ('.png' if kind == 'target' else '.npz'), 'wrong asset type')
            require(sha256(p) == item[kind + '_sha256'], 'record asset hash mismatch')
            assets[str(p)] = item[kind + '_sha256']
            paths[kind] = p
        masks = bounded_npz(paths['masks'], 16, {'allowed', 'protected', 'uncertain', 'target_teeth'})
        # Pixel checks (including closed target equality) occur before model import in worker.
        frames.append((item, paths, masks))
    require(seen == OPEN | CLOSED | CONTACT, 'require all three open, three closed and one protected-contact control')
    return frames, assets


def elapsed_charge(snapshot, report, base, output):
    """Snapshot report and each OTHER supervisor once, including failed attempts."""
    charge = float(report['wall_seconds'])
    require(math.isfinite(charge) and 0 <= charge <= 600, 'invalid snapshot wall time')
    for path in base.rglob('supervisor.json') if base.exists() else ():
        if path.parent.resolve() in (snapshot.resolve(), output.resolve()):
            continue
        seconds = float(read_json(path)['wall_seconds'])
        require(math.isfinite(seconds) and seconds >= 0, 'invalid supervisor wall time')
        charge += seconds
    require(charge < 1800, 'cumulative 1800s exhausted')
    return charge


def verify_snapshot(snapshot, workspace):
    report = read_json(snapshot / 'report.json')
    require(report.get('status') == 'completed' and report.get('supervisor_verified') is True, 'snapshot not verified')
    require(re.fullmatch('[0-9a-f]{40}', report['code_sha']), 'invalid snapshot code SHA')
    files = {str(snapshot / 'report.json'): sha256(snapshot / 'report.json')}
    for name, digest in report['files'].items():
        p = record_file(name, snapshot, workspace)
        require(sha256(p) == digest, 'snapshot asset hash mismatch: ' + name)
        files[str(p)] = digest
    required = {'source_canvas.png', 'source_crop.npz'}
    required.update(f'{prefix}_f{frame:04d}.{suffix}' for frame in OPEN | CLOSED | CONTACT
                    for prefix, suffix in (('H', 'npz'), ('raw512', 'png')))
    require(required <= set(report['files']), 'snapshot manifest omits required assets')
    for key in ('weights', 'inputs'):
        require(report[key + '_before'] == report[key + '_after'] and report[key + '_before'], key + ' changed')
        for name, digest in report[key + '_before'].items():
            p = inside(name, workspace)
            require(sha256(p) == digest, key + ' hash mismatch')
            files[str(p)] = digest
    for name in ('src/modules/spade_generator.py', 'src/modules/util.py'):
        require(command('git', 'rev-parse', 'HEAD:' + name) ==
                command('git', 'rev-parse', report['code_sha'] + ':' + name), 'snapshot network code differs')
    return report, files


def unchanged(files):
    require(all(sha256(p) == h for p, h in files.items()), 'immutable input/weight/target changed')


def worker(args, workspace, snapshot, base, output, report, files, frames):
    started = time.monotonic()
    from PIL import Image
    loaded = []
    for item, paths, masks in frames:
        raw = read_raw_png(snapshot / f"raw512_f{item['frame']:04d}.png")
        target = read_raw_png(paths['target'])
        validate_pixels(raw, target, masks, item['role'])
        h = bounded_npz(snapshot / f"H_f{item['frame']:04d}.npz", 12, {'H'})['H']
        require(h.shape == (1, 64, 256, 256) and h.dtype == np.float16 and np.isfinite(h).all(), 'invalid production H')
        require(array_hash(h) == report['selected_H'][str(item['frame'])]['array_sha256'], 'H array mismatch')
        loaded.append((item, masks, raw.copy(), target.copy(), h))

    import torch
    import cv2
    import yaml
    from src.modules.spade_generator import SPADEDecoder
    from src.modules.upper_teeth_adapter import safe_feature_mask
    from src.utils.crop import prepare_paste_back, paste_back

    env = report['environment']
    require(env['deterministic_algorithms'] is True and env['tf32'] is False and
            env['cudnn_benchmark'] is False and report['cfg']['flag_use_half_precision'] is True,
            'unsupported snapshot numerical environment')
    require(str(torch.__version__) == env['torch'] and torch.version.cuda == env['cuda'] and
            cv2.__version__ == env['cv2'] and np.__version__ == env['numpy'], 'snapshot runtime version mismatch')
    torch.manual_seed(env['seed'])
    np.random.seed(env['seed'])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.device_count() == 1, 'one visible GPU required')
    config = inside(report['cfg']['models_config'], workspace)
    files[str(config)] = sha256(config)
    # Config must be the same Git blob too; no architecture guessing from a checkpoint.
    rel = 'src/config/models.yaml'
    require(command('git', 'hash-object', str(config)) == command('git', 'rev-parse', report['code_sha'] + ':' + rel)
            == command('git', 'rev-parse', 'HEAD:' + rel), 'model config changed')
    model = SPADEDecoder(**yaml.safe_load(config.read_text())['model_params']['spade_generator_params']).cuda()
    checkpoint = inside(report['cfg']['checkpoint_G'], workspace)
    require(str(checkpoint) in report['weights_before'], 'G not in snapshot weight manifest')
    model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
    model.eval().requires_grad_(False)

    def states():
        return {kind: {n: array_hash(t.detach().cpu().numpy()) for n, t in values}
                for kind, values in [('parameters', model.named_parameters()), ('buffers', model.named_buffers())]}

    state_before = states()
    canvas = np.asarray(Image.open(snapshot / 'source_canvas.png')).copy()
    matrix = bounded_npz(snapshot / 'source_crop.npz', 16)['M_c2o']
    require(matrix.shape in ((2, 3), (3, 3)) and np.isfinite(matrix).all(), 'invalid crop matrix')
    mask_path = inside(ROOT / 'src/utils/resources/mask_template.png', workspace)
    files[str(mask_path)] = sha256(mask_path)
    crop_mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    require(crop_mask is not None and array_hash(crop_mask) == report['cfg']['mask_crop']['array_sha256'], 'original crop mask mismatch')
    size = (canvas.shape[1], canvas.shape[0])
    paste_mask = prepare_paste_back(crop_mask, matrix, size)

    def decode(hidden):
        with torch.autocast('cuda', dtype=torch.float16):
            return model.decode_features(hidden).float()

    def rgb(value):
        return (value.detach().cpu().numpy()[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)

    results = []
    first_trial_timing = None
    for item, masks, raw, target, hidden in loaded:
        budget(workspace, base)
        frame = item['frame']
        folder = output / f'f{frame:04d}'
        folder.mkdir()
        h = torch.from_numpy(hidden).cuda()
        allowed = torch.from_numpy(masks['allowed'][None, None]).cuda()
        safe = safe_feature_mask(allowed, allowed.float())
        safe_count = int(safe.sum().item())
        # Distinguish limited safe support from failure of the reachable tail features.
        reachable = torch.nn.functional.max_pool2d(safe.float(), 3, stride=1, padding=1)
        reachable = reachable.repeat_interleave(2, 2).repeat_interleave(2, 3).bool()
        scale = ((h.float().square() * safe).sum((2, 3), keepdim=True) / max(1, safe_count)).sqrt().clamp_min(1e-4)
        teacher = torch.from_numpy(target.transpose(2, 0, 1).copy()[None]).cuda().float() / 255
        with torch.no_grad():
            original = decode(h)
        require(np.array_equal(rgb(original), raw), 'production raw512 reproduction failed')
        full_before = paste_back(raw, matrix, canvas, paste_mask)
        domain = cv2.warpAffine(masks['allowed'].astype(np.float32), matrix[:2], size,
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0
        baseline_mse = float(((original - teacher).square() * allowed).sum().item() / max(1, int(masks['allowed'].sum()) * 3))

        def measure(prediction, delta):
            require(torch.isfinite(prediction).all().item(), 'numerical_failure: nonfinite decode')
            float_outside = float((prediction - original).abs().masked_select(~allowed.expand_as(prediction)).max().item())
            learned = rgb(prediction)
            outside = int(np.abs(learned.astype(np.int16) - raw.astype(np.int16))[~masks['allowed']].max(initial=0))
            protected = int(np.abs(learned.astype(np.int16) - raw.astype(np.int16))[masks['protected']].max(initial=0))
            full = paste_back(learned, matrix, canvas, paste_mask)
            full_outside = int(np.abs(full.astype(np.int16) - full_before.astype(np.int16))[~domain].max(initial=0))
            require(float_outside == outside == protected == full_outside == 0, 'protection leakage: stop immediately')
            actual = ((h + delta).float() - h.float()) / scale
            normalized = actual.masked_select(safe.expand_as(actual))
            require(torch.isfinite(actual).all().item(), 'numerical_failure: nonfinite actual normalized residual')
            stats = {'raw_outside_float_max': float_outside, 'raw_outside_uint8_max': outside,
                     'protected_lower_lip_uint8_max': protected, 'full_propagation_outside_uint8_max': full_outside,
                     'normalized_residual_rms': float(normalized.square().mean().sqrt().item()) if normalized.numel() else 0.,
                     'normalized_residual_max': float(normalized.abs().max().item()) if normalized.numel() else 0.}
            check_actual_radius(float(actual.abs().max().item()), radius)
            stats.update(nominal_radius=radius, normalized_absolute_tolerance=NORMALIZED_TOLERANCE)
            return stats, learned, full

        def loss_terms(prediction, delta):
            mse = ((prediction - teacher).square() * allowed).sum() / (allowed.sum().clamp_min(1) * 3)
            edge = prediction.new_zeros(())
            smooth = prediction.new_zeros(())
            normalized = delta.float() / scale
            for axis in (2, 3):
                a, b = [slice(None)] * 4, [slice(None)] * 4
                a[axis], b[axis] = slice(1, None), slice(None, -1)
                a, b = tuple(a), tuple(b)
                pair = allowed[a] & allowed[b]
                error = (prediction[a] - prediction[b]) - (teacher[a] - teacher[b])
                edge = edge + (error.square() * pair).sum() / (pair.sum().clamp_min(1) * 3)
                smooth = smooth + (normalized[a] - normalized[b]).square().mean()
            total = mse + .1 * edge + 1e-5 * normalized.square().mean() + 1e-5 * smooth
            return total, mse, edge

        trials, best = [], None
        for radius in RADII if item['role'] == 'open' else (0.,):
            z = torch.zeros_like(h, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=.08)
            scaler = torch.cuda.amp.GradScaler(init_scale=256)
            history = []
            any_actual_change = False
            trial_started = time.monotonic()
            steps = 200 if safe_count and radius else 0
            trace_path = folder / f'r{radius:.2f}_steps.jsonl'
            for step in range(steps + 1):
                step_started = time.monotonic()
                delta = (radius * z.tanh() * scale * safe).to(h.dtype)
                require(torch.isfinite(delta).all().item() and
                        (h.float() + delta.float()).abs().max().item() <= torch.finfo(torch.float16).max / 2,
                        'numerical_failure: half-float safety limit')
                prediction = decode(h + delta)
                total, mse, edge = loss_terms(prediction, delta)
                require(torch.isfinite(total).item(), 'numerical_failure: nonfinite loss')
                # Every iterate is checked, not only the final selected trial.
                stats, learned, full = measure(prediction, delta)
                any_actual_change |= bool(np.any(learned != raw))
                loss_scale = float(scaler.get_scale())
                gradient = gradient_rms = None
                gradient_nonzero = 0
                if step < steps:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    require(z.grad is not None and torch.isfinite(z.grad).all().item(),
                            f'numerical_failure: nonfinite/missing unscaled gradient at frame={frame} radius={radius} step={step} scale={loss_scale}; not a capacity verdict')
                    gradient = float(z.grad.abs().max().item())
                    gradient_rms = float(z.grad.square().mean().sqrt().item())
                    gradient_nonzero = int(torch.count_nonzero(z.grad).item())
                    require(gradient_nonzero > 0,
                            f'optimization_failure: all-zero unscaled gradient at frame={frame} radius={radius} step={step} scale={loss_scale} grad_max={gradient}; not a capacity verdict')
                    scaler.step(optimizer)
                    scaler.update()
                torch.cuda.synchronize()
                step_seconds = time.monotonic() - step_started
                history.append({'step': step, 'loss': float(total.item()), 'mse': float(mse.item()),
                                'teacher_neighbor_mse': float(edge.item()), 'loss_scale': loss_scale,
                                'unscaled_gradient_abs_max': gradient, 'unscaled_gradient_rms': gradient_rms,
                                'unscaled_gradient_nonzero': gradient_nonzero, 'step_seconds': step_seconds, **stats})
                if steps and first_trial_timing is None and step == min(9, steps):
                    average = (time.monotonic() - trial_started) / (step + 1)
                    first_trial_timing = {'frame': frame, 'radius': radius, 'sampled_steps': step + 1,
                                          'seconds_per_step_including_full_protection': average,
                                          'estimated_9_open_trials_seconds': average * 201 * 9,
                                          'estimate_not_guarantee': True, 'protection_checks_not_reduced': True}
                    (output / 'first_trial_timing.json').write_text(json.dumps(first_trial_timing, indent=2) + '\n')
                with trace_path.open('a', encoding='utf-8') as trace:
                    trace.write(json.dumps(history[-1]) + '\n')
                if step % 10 == 0:
                    budget(workspace, base)
            require(not steps or any_actual_change,
                    'optimization_failure: 200 steps without actual raw512 change; not a capacity verdict')
            prefix = f'r{radius:.2f}'
            for name, image in [('before_raw512', raw), ('target_raw512', target), ('learned_raw512', learned),
                                ('before_full', full_before), ('target_full', paste_back(target, matrix, canvas, paste_mask)),
                                ('learned_full', full)]:
                Image.fromarray(image).save(folder / f'{prefix}_{name}.png')
            parts = {}
            xs = np.where(masks['allowed'])[1]
            cuts = np.linspace(xs.min(), xs.max() + 1, 4).astype(int) if xs.size else [0, 171, 342, 512]
            for i, name in enumerate(('left', 'mid', 'right')):
                region = masks['allowed'].copy()
                region[:, :cuts[i]] = False
                region[:, cuts[i + 1]:] = False
                parts[name] = {'changed_pixels': int(np.any(learned != raw, axis=2)[region].sum()),
                               'target_uint8_mse_before': float(np.square(raw[region].astype(float) - target[region]).mean()) if region.any() else None,
                               'target_uint8_mse_after': float(np.square(learned[region].astype(float) - target[region]).mean()) if region.any() else None}
            teeth = masks['target_teeth']
            boundary = teeth & ~cv2.erode(teeth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            with torch.no_grad():
                final_image = decode(h + delta)
                def domain_mse(image, domain):
                    return float(((image-teacher).square()*domain).sum().item() / max(1, int(domain.sum().item())*3))
                support_metrics = {'reachable_pixels': int(reachable.sum().item()),
                    'reachable_mse_before': domain_mse(original, reachable),
                    'reachable_mse_after': domain_mse(final_image, reachable),
                    'unreachable_target_pixels': int((allowed & ~reachable).sum().item()),
                    'unreachable_mse_before': domain_mse(original, allowed & ~reachable),
                    'unreachable_mse_after': domain_mse(final_image, allowed & ~reachable)}
            trial = {'radius': radius, 'steps': steps, 'safe_feature_cells': safe_count, 'history': history,
                     'support_metrics': support_metrics,
                     'baseline_float_mse': baseline_mse, 'final_float_mse': float(mse.item()), 'protection': stats,
                     'parts': parts, 'target_teeth_boundary_rgb_mae_proxy_not_visual_acceptance':
                     float(np.abs(learned.astype(float) - target)[boundary].mean()) if boundary.any() else None}
            trials.append(trial)
            if best is None or trial['final_float_mse'] < best[0]:
                best = (trial['final_float_mse'], radius, delta.detach().cpu().numpy().copy(), learned.copy())
            del optimizer, z, prediction, total, mse, edge
        np.savez(folder / 'best_delta_NOT_DEPLOYABLE.npz', delta=best[2])
        with np.load(folder / 'best_delta_NOT_DEPLOYABLE.npz', allow_pickle=False) as saved:
            with torch.no_grad():
                reloaded = decode(h + torch.from_numpy(saved['delta']).cuda())
        require(np.array_equal(rgb(reloaded), best[3]), 'saved delta exact replay failed')
        results.append({'frame': frame, 'role': item['role'], 'trials': trials, 'selected_radius': best[1],
                        'delta_replay_raw_exact': True, 'selection': 'minimum final float masked MSE; display only, NOT acceptance'})
        (folder / 'metrics.json').write_text(json.dumps(results[-1], indent=2) + '\n')
    state_after = states()
    require(state_before == state_after, 'frozen parameters/buffers changed')
    unchanged(files)
    result = {'status': 'completed', 'purpose': 'diagnostic_counterfactual_only', 'not_deployable': True,
              'visual_acceptance': False, 'human_semantic_mask_approved': False,
              'normalized_absolute_tolerance': NORMALIZED_TOLERANCE, 'first_trial_timing': first_trial_timing,
              'amp_scaling': 'GradScaler init_scale=256; unscaled gradients checked; no convergence guarantee',
              'counterfactual_is_gt': False, 'shared_model': False, 'video_generated': False,
              'protection_scope': 'raw allowed and bilinear propagated domain only; not complete semantic upper-teeth guarantee',
              'code_sha': code_check(), 'snapshot_code_sha': report['code_sha'], 'environment': env,
              'execution_gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'frames': results,
              'base_before': state_before, 'base_after': state_after, 'immutable_files': files,
              'wall_seconds': time.monotonic() - started, 'files':
              {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print('E1 native keyframes (no video, no visual acceptance):', output, flush=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'snapshot', 'region-record', 'output', 'budget-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--authorize-e1', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform == 'linux' and args.authorize_e1, 'Linux and explicit E1 authorization required')
    workspace = Path(args.workspace).resolve(strict=True)
    inside(ROOT, workspace)
    snapshot = inside(args.snapshot, workspace)
    record = inside(args.region_record, workspace)
    base = inside(args.budget_root, workspace, exists=False)
    output = inside(args.output, workspace, exists=False)
    require(output != base and output.is_relative_to(base), 'output must be below budget-root')
    require(not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'budget root separate from code')
    require(not snapshot.is_relative_to(output) and not record.is_relative_to(output), 'output overlaps inputs')
    require(snapshot.is_relative_to(base), 'snapshot must share cumulative budget-root')
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)) and inside(os.environ[key], workspace).is_dir(), 'missing isolated cache ' + key)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'one explicit GPU UUID required')
    code = code_check()
    report, files = verify_snapshot(snapshot, workspace)
    frames, assets = validate_record(record, snapshot, report, workspace)
    files.update(assets)
    charge = elapsed_charge(snapshot, report, base, output)
    limit = min(1200., 1800. - charge)
    if args._worker:
        require(os.environ.get('UPPER_E1_PARENT') == str(os.getppid()), 'worker must be supervised')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        try:
            worker(args, workspace, snapshot, base, output, report, files, frames)
        except Exception as exc:
            reason = str(exc)
            category = next((k for k in ('numerical_failure', 'optimization_failure') if k in reason), 'execution_failure')
            (output / 'failure.json').write_text(json.dumps({'status': category, 'reason': reason,
                'not_deployable': True, 'capacity_verdict': None, 'visual_acceptance': False}) + '\n')
            raise
        return
    require(not output.exists() and not Path(args.output).is_symlink(), 'new output only; never delete evidence')
    initial = budget(workspace, base)
    require(time.monotonic() - started < limit, 'guards exhausted time allowance')
    output.mkdir(parents=True)
    env = dict(os.environ, UPPER_E1_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                               cwd=ROOT, env=env, start_new_session=True)
    owned = OwnedProcessGroup(process)
    timer = threading.Timer(max(0, limit - (time.monotonic() - started)), owned.kill)
    timer.daemon = True
    timer.start()
    try:
        while not owned.exited():
            require(time.monotonic() - started < limit, 'E1/cumulative hard timeout')
            budget(workspace, base)
            time.sleep(.5)
        owned.finish()
        require(process.returncode == 0, 'E1 worker failed; partial evidence is NOT accepted')
        path = output / 'report.json'
        result = read_json(path)
        require(result['status'] == 'completed' and result['not_deployable'] is True and result['visual_acceptance'] is False, 'invalid worker report')
        for name, digest in result['files'].items():
            require(sha256(record_file(name, output, workspace)) == digest, 'output artifact hash mismatch')
        unchanged(result['immutable_files'])
        require(code_check() == code, 'code changed')
        seconds = time.monotonic() - started
        require(seconds < limit, 'verification exceeded time allowance')
        result.update(supervisor_verified=True, wall_seconds=seconds, prior_wall_seconds=charge,
                      cumulative_wall_seconds=charge + seconds, hard_limit_seconds=limit,
                      budget_before=initial, budget_after=budget(workspace, base))
        path.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        timer.cancel()
        owned.finish()
        (output / 'supervisor.json').write_text(json.dumps({'returncode': process.returncode,
            'wall_seconds': time.monotonic() - started, 'hard_limit_seconds': limit,
            'prior_wall_seconds': charge}) + '\n')


if __name__ == '__main__':
    main()
