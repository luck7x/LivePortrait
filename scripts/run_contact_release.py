"""Bounded contact-boundary experiment. Train labels never enter render prediction."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_contact_samples as prep
from scripts.snapshot_upper_teeth import (OwnedProcessGroup, array_hash, audio_signature,
    budget, check_video, code_check, command, inside, media, require, sha256)
from scripts.probe_upper_tail_capacity import bounded_npz

SHAPE = (1, 1, 50, 120)
ARCH = 'contact-boundary-pointwise-v1'
LABEL_KEYS = {'allowed', 'target_delta', 'uncertain', 'protected'}


def write_json(path, value):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path, limit=2**20):
    require(path.is_file() and path.stat().st_size <= limit, 'JSON size/type limit')
    return json.loads(path.read_text(encoding='utf-8'))


def tracked(path, root, manifest, expected=None):
    path = inside(path, root)
    require(path.is_file(), 'input must be a file')
    actual = sha256(path)
    require(expected is None or actual == expected, 'file hash mismatch: ' + str(path))
    manifest[str(path)] = actual
    return path


def sample_metadata(directory, snapshot, report, workspace, manifest):
    path = tracked(directory / 'report.json', workspace, manifest)
    meta = read_json(path, 32 * 2**20)
    require(meta['status'] == 'completed' and meta['supervisor_verified'] is True and
            meta['base_frozen'] is True and meta['base_eval'] is True and
            meta['base_before'] == meta['base_after'] and
            meta['inputs_before'] == meta['inputs_after'], 'samples not verified')
    require(meta['snapshot_code_sha'] == report['code_sha'] and
            Path(meta['snapshot']).resolve() == snapshot and
            meta['sourceF_K_hashes'] == report['snapshot_arrays'] and
            meta['inputweight_hashes'] == report['weights_before'] and
            meta['ROI'] == list(prep.ROI) and meta['frame_list'] == prep.sample_indices(),
            'samples/snapshot provenance mismatch')
    require(re.fullmatch('[0-9a-f]{40}', meta['current_code']), 'invalid samples SHA')
    require([v['frame'] for v in meta['frames']] == prep.sample_indices(), 'sample records mismatch')
    for item in meta['frames']:
        f = item['frame']
        require(item['previous'] == prep.predecessor(f) and
                item['raw512'] == report['frames'][f]['raw512'] and
                item['final_k'] == report['frames'][f]['final_k'] and
                item['previous_k'] == report['frames'][prep.predecessor(f)]['final_k'],
                'sample frame provenance mismatch')
    # Render reads this metadata only: no feature NPZ, annotation or raw image opens.
    return meta


def validate_label_arrays(arrays):
    require(set(arrays) == LABEL_KEYS, 'label keys mismatch')
    for key, value in arrays.items():
        require(value.shape == SHAPE and value.dtype == (np.float32 if key == 'target_delta' else np.bool_),
                'label shape/dtype mismatch: ' + key)
    a, u, p, d = (arrays[k] for k in ('allowed', 'uncertain', 'protected', 'target_delta'))
    require(not (a & (u | p)).any(), 'allowed overlaps protected/uncertain')
    require(np.isfinite(d).all() and (d >= -.6).all() and (d <= 0).all() and
            (d[~a] == 0).all(), 'invalid target delta')


def load_labels(directory, samples, meta, manifest):
    path = tracked(directory / 'record.json', directory, manifest)
    record = read_json(path)
    require(record['schema'] == 'contact-boundary-counterfactual-v1' and
            record['purpose'] == 'limited-source-boundary-experiment' and
            record['user_authorized_experiment'] is True and
            record['human_semantic_mask_approved'] is False and
            record['independent_review'] == 'passed' and record['ROI'] == list(prep.ROI),
            'labels not approved for this limited diagnostic experiment')
    require(record['samples_code_sha'] == meta['current_code'] and
            record['samples_report_sha256'] == sha256(samples / 'report.json'), 'labels provenance mismatch')
    rows = record['frames']
    require(len(rows) == len(prep.sample_indices()) and
            sorted(r['frame'] for r in rows) == prep.sample_indices(), 'one label per sample required')
    labels, splits = {}, {}
    for row in rows:
        frame = row['frame']
        require(type(frame) is int and row['split'] in ('train', 'validation'), 'invalid split/frame')
        require(row['file'] == f'labels_f{frame:04d}.npz', 'label must be exact basename')
        # raw_sha256 is the file SHA256 of the published PNG, not array_hash(raw).
        require(row['raw_sha256'] == meta['files'][f'raw512_f{frame:04d}.png'], 'label source raw hash mismatch')
        path = tracked(directory / row['file'], directory, manifest, row['sha256'])
        value = bounded_npz(path, 1, LABEL_KEYS)
        validate_label_arrays(value)
        labels[frame], splits[frame] = value, row['split']
    train = sorted(f for f in splits if splits[f] == 'train')
    val = sorted(f for f in splits if splits[f] == 'validation')
    require({225, 226, 227, 228, 251, 252} <= set(train) and
            set(range(254, 260)) | {446, 456} <= set(val), 'required train/validation coverage missing')
    for f in (225, 251, 252, 292):
        require(not labels[f]['allowed'].any() and not labels[f]['target_delta'].any(),
                'contact/closed control must be negative')
    require(any(labels[f]['allowed'].any() for f in train), 'no positive training frames')
    return labels, train, val


def elapsed_charge(base):
    total = 0.
    for path in sorted(base.rglob('supervisor.json')) if base.exists() else []:
        inside(path, base)
        row = read_json(path)
        require(row.get('status') in ('completed', 'failed'), 'unresolved supervisor; audit before resuming')
        seconds = row.get('wall_seconds')
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                'invalid prior wall charge')
        total += seconds
    require(total <= 1800, 'cumulative 1800s budget exhausted')
    return total


def fixed_batch(step, train, positives):
    """Seeded frame *order*, never frame IDs/coordinates as model input."""
    rng = np.random.default_rng(1729)
    rest = list(rng.permutation(train))
    pos = list(rng.permutation(positives))
    return [int(pos[step % len(pos)])] + [int(rest[(3 * step + j) % len(rest)]) for j in range(3)]


def pixel_audit(base, candidate, support):
    require(base.shape == candidate.shape and support.shape == base.shape[:2], 'audit shape mismatch')
    diff = np.abs(candidate.astype(np.float64) - base.astype(np.float64))
    outside = diff[~support]
    return {'changed_pixels': int(np.any(diff != 0, axis=-1).sum()),
            'outside_pixels': int(np.any(outside != 0, axis=-1).sum()),
            'outside_max': float(outside.max(initial=0))}


def tensor_image(value):
    a = value.detach().float().cpu().numpy()[0].transpose(1, 2, 0)
    return a, (a.clip(0, 1) * 255).astype(np.uint8)


def runtime(report):
    import torch
    env = report['environment']
    require(env['deterministic_algorithms'] is True and env['tf32'] is False and
            env['cudnn_benchmark'] is False and report['cfg']['flag_use_half_precision'] is True and
            report['cfg']['flag_do_torch_compile'] is False, 'unsupported numerical config')
    require(str(torch.__version__) == env['torch'] and torch.version.cuda == env['cuda'] and
            np.__version__ == env['numpy'], 'snapshot runtime mismatch')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS configuration mismatch')
    torch.manual_seed(env['seed'])
    np.random.seed(env['seed'])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.device_count() == 1, 'exactly one visible GPU required')
    torch.cuda.reset_peak_memory_stats()
    return torch


def head_from_checkpoint(path, workspace, snapshot_report, samples_meta, manifest):
    import torch
    from src.modules.contact_boundary_head import ContactBoundaryHead
    path = tracked(path, workspace, manifest)
    require(path.stat().st_size <= 2**20, 'head checkpoint size limit')
    metadata_path = tracked(path.with_suffix('.json'), workspace, manifest)
    meta = read_json(metadata_path)
    require(meta['arch'] == ARCH and meta['checkpoint_sha256'] == sha256(path) and
            meta['ROI'] == list(prep.ROI) and meta['threshold'] == .9 and meta['max_logit_delta'] == .6 and
            meta['snapshot_arrays'] == snapshot_report['snapshot_arrays'] and
            meta['weights'] == snapshot_report['weights_before'] and
            meta['samples_code_sha'] == samples_meta['current_code'] and
            meta['samples_files'] == samples_meta['files'] and
            meta['code_sha'] == code_check(), 'checkpoint provenance mismatch')
    require(meta['head_blob'] == command('git', 'hash-object', str(ROOT / 'src/modules/contact_boundary_head.py')),
            'checkpoint architecture code mismatch')
    train, val = meta['train_frames'], meta['validation_frames']
    require(len(train) + len(val) == len(prep.sample_indices()) and not set(train) & set(val) and
            sorted(train + val) == prep.sample_indices(), 'checkpoint splits invalid')
    state = torch.load(path, weights_only=True, map_location='cuda')
    require(isinstance(state, dict) and 'channel_scale' in state and
            all(isinstance(t, torch.Tensor) and torch.isfinite(t).all().item() for t in state.values()),
            'invalid head state')
    head = ContactBoundaryHead(torch.ones((1, 32, 1, 1), device='cuda'))
    head.load_state_dict(state, strict=True)
    require(all(t.dtype == torch.float32 for t in state.values()), 'head checkpoint must be FP32')
    head.eval().requires_grad_(False)
    return head, meta


def gate_metrics(result, label):
    active = result['gate'].detach().cpu().numpy() > 0
    allowed = label['allowed']
    tp = int((active & allowed).sum())
    return {'positive_pixels': int(active.sum()), 'true_positive': tp,
            'precision': tp / max(1, int(active.sum())), 'recall': tp / max(1, int(allowed.sum())),
            'protected_pred_active': int((active & label['protected']).sum()),
            'uncertain_pred_active': int((active & label['uncertain']).sum()),
            'outside_allowed_pred_active': int((active & ~allowed).sum())}


def train_stage(args, workspace, snapshot, base, output, report, arrays, meta, manifest):
    labels_dir = inside(args.labels, workspace)
    labels, train, val = load_labels(labels_dir, Path(args.samples).resolve(), meta, manifest)
    features, logits = {}, {}
    for f in prep.sample_indices():
        name = f'sample_f{f:04d}.npz'
        path = tracked(Path(args.samples) / name, workspace, manifest, meta['files'][name])
        sample = bounded_npz(path, 2, {'feature', 'logits'})
        require(prep.validate_sample(sample) == meta['arrayhashes'][name], 'sample array mismatch')
        features[f], logits[f] = sample['feature'], sample['logits']
    write_json(output / 'report.json', {'status': 'running', 'inputs_before': manifest,
                                       'supervisor_verified': False, 'stage': 'train'})
    torch = runtime(report)
    from PIL import Image
    from src.modules.contact_boundary_head import ContactBoundaryHead
    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    # Only training features contribute to normalization. Validation is post-hoc only.
    scale = np.sqrt(np.mean(np.stack([features[f].astype(np.float64) ** 2 for f in train]),
                            axis=(0, 3, 4), keepdims=False)).astype(np.float32).reshape(1, 32, 1, 1)
    head = ContactBoundaryHead(torch.from_numpy(scale).cuda()).train()
    x = {f: torch.from_numpy(v).cuda() for f, v in features.items()}
    y = {f: {k: torch.from_numpy(v).cuda() for k, v in label.items()} for f, label in labels.items()}
    positives = [f for f in train if labels[f]['allowed'].any()]
    positive = sum(int(labels[f]['allowed'].sum()) for f in train)
    negative = len(train) * 6000 - positive
    posweight = min(50., negative / positive)
    log, failures = [], []
    delta_initial = {n: p.detach().clone() for n, p in head.delta_head.named_parameters()}
    opt = torch.optim.Adam(list(head.backbone.parameters()) + list(head.gate_head.parameters()), lr=.004)
    updates = {'gate': 0, 'delta': 0}

    def optimize(loss, optimizer, parameters, stage):
        require(torch.isfinite(loss).item(), 'numerical_failure: nonfinite loss')
        old = [p.detach().clone() for p in parameters]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in parameters]
        require(all(g is not None and torch.isfinite(g).all().item() for g in grads), 'numerical_failure: nonfinite/missing gradient')
        require(any(torch.count_nonzero(g).item() for g in grads), 'numerical_failure: zero gradient')
        optimizer.step()
        require(all(torch.isfinite(p).all().item() for p in parameters), 'numerical_failure: nonfinite parameter')
        require(any(not torch.equal(a, b) for a, b in zip(old, parameters)), 'numerical_failure: no update')
        updates[stage] += 1

    def progress(stage, step, loss):
        values = [gate_metrics(head(x[f]), labels[f]) for f in train]
        item = {'stage': stage, 'step': step, 'loss': float(loss.detach()), 'train_gate_metrics': values}
        log.append(item)
        print(json.dumps(item), flush=True)

    try:
        for step in range(600):
            batch = fixed_batch(step, train, positives)
            xx = torch.cat([x[f] for f in batch])
            yy = torch.cat([y[f]['allowed'] for f in batch]).float()
            result = head(xx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(result['gate_logits'], yy,
                        pos_weight=torch.tensor(posweight, device='cuda'))
            optimize(loss, opt, list(head.backbone.parameters()) + list(head.gate_head.parameters()), 'gate')
            require(all(torch.equal(delta_initial[n], p) and not torch.count_nonzero(p).item()
                        for n, p in head.delta_head.named_parameters()), 'stage A changed zero delta')
            if (step + 1) % 100 == 0:
                progress('A', step + 1, loss)
                budget(workspace, base)
        with torch.no_grad():
            covered = {f: bool(((head(x[f])['gate'] > 0) & y[f]['allowed']).any().item()) for f in positives}
        require(all(covered.values()), 'coverage_failure: gate misses a positive training frame')
        head.freeze_gate()
        frozen = {n: p.detach().clone() for n, p in head.named_parameters() if not p.requires_grad}
        opt = torch.optim.Adam(head.delta_head.parameters(), lr=.01)
        for step in range(300):
            batch = fixed_batch(step, train, positives)
            xx = torch.cat([x[f] for f in batch])
            allowed = torch.cat([y[f]['allowed'] for f in batch])
            target = torch.cat([y[f]['target_delta'] for f in batch])
            correction = head(xx)['correction']  # predicted gate; never teacher-gated
            loss = ((correction - target).square()[allowed].mean() +
                    .1 * correction.square()[~allowed].mean())
            optimize(loss, opt, list(head.delta_head.parameters()), 'delta')
            require(all(torch.equal(frozen[n], p) for n, p in head.named_parameters() if n in frozen),
                    'stage B changed frozen gate/backbone')
            if (step + 1) % 100 == 0:
                progress('B', step + 1, loss)
                budget(workspace, base)
    except RuntimeError as exc:
        # Only diagnosed coverage/numerical failure is a saved, explicitly failed candidate.
        if not any(key in str(exc) for key in ('coverage_failure:', 'numerical_failure:')):
            raise
        failures.append(str(exc))
    head.eval()
    checkpoint = output / 'candidate_checkpoint.pt'
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, checkpoint)
    checkpoint_meta = {'arch': ARCH, 'checkpoint_sha256': sha256(checkpoint), 'ROI': list(prep.ROI),
        'threshold': .9, 'max_logit_delta': .6, 'train_frames': train, 'validation_frames': val,
        'snapshot_arrays': report['snapshot_arrays'], 'weights': report['weights_before'],
        'samples_code_sha': meta['current_code'], 'samples_files': meta['files'],
        'samples_report_sha256': sha256(Path(args.samples) / 'report.json'),
        'label_record_sha256': sha256(labels_dir / 'record.json'),
        'label_files': {Path(k).name: v for k, v in manifest.items() if Path(k).is_relative_to(labels_dir)},
        'code_sha': code_check(), 'head_blob': command('git', 'hash-object', str(ROOT / 'src/modules/contact_boundary_head.py')),
        'purpose': 'limited counterfactual boundary darkening; not native GT',
        'human_semantic_mask_approved': False, 'quality_pass': False}
    write_json(checkpoint.with_suffix('.json'), checkpoint_meta)
    # Output files are audited separately from immutable inputs.
    reload_head, _ = head_from_checkpoint(checkpoint, workspace, report, meta, {})
    evaluation = []
    with torch.no_grad():
        for f in prep.sample_indices():
            r = head(x[f])
            rr = reload_head(x[f])
            require(all(torch.equal(r[k], rr[k]) for k in r), 'checkpoint prediction reload mismatch')
            original = torch.from_numpy(logits[f]).cuda().half()
            base_tensor = torch.sigmoid(original)
            candidate = torch.sigmoid(original + r['correction'].half())
            require(torch.equal(candidate, torch.sigmoid(original + rr['correction'].half())), 'reload raw mismatch')
            bfloat, braw = tensor_image(base_tensor)
            cfloat, craw = tensor_image(candidate)
            support = r['gate'].cpu().numpy()[0, 0] > 0
            numerical = pixel_audit(bfloat, cfloat, support)
            pixels = pixel_audit(braw, craw, support)
            require(numerical['outside_max'] == pixels['outside_max'] == 0, 'pointwise support leakage')
            metrics = gate_metrics(r, labels[f])
            semantic = {k: pixel_audit(braw, craw, ~labels[f][k][0, 0])
                        for k in ('protected', 'uncertain')}
            semantic['outside_allowed'] = pixel_audit(braw, craw, labels[f]['allowed'][0, 0])
            semantic_float = {k: pixel_audit(bfloat, cfloat, ~labels[f][k][0, 0])
                              for k in ('protected', 'uncertain')}
            semantic_float['outside_allowed'] = pixel_audit(bfloat, cfloat, labels[f]['allowed'][0, 0])
            safe = all(v['outside_max'] == 0 for v in [*semantic.values(), *semantic_float.values()])
            coverage = not labels[f]['allowed'].any() or metrics['recall'] == 1.
            if not safe or not coverage or metrics['protected_pred_active'] or metrics['uncertain_pred_active']:
                failures.append(f'frame {f}: semantic safety/coverage failed')
            np.savez_compressed(output / f'prediction_f{f:04d}.npz',
                predicted_correction=r['correction'].cpu().numpy(), gate=r['gate'].cpu().numpy(),
                probability=r['probability'].cpu().numpy())
            Image.fromarray(craw).save(output / f'predictedROI_f{f:04d}.png')
            evaluation.append({'frame': f, 'split': 'train' if f in train else 'validation',
                **metrics, 'outside_predgate_float': numerical, 'outside_predgate_uint8': pixels,
                'semantic_uint8': semantic, 'semantic_float': semantic_float,
                'rawROI_sha256': array_hash(craw), 'reload_exact': True})
    gpu_end.record()
    torch.cuda.synchronize()
    checkpoint_meta.update(quality_pass=not failures, failures=failures, checkpoint_reload_exact=True)
    write_json(checkpoint.with_suffix('.json'), checkpoint_meta)
    return {'stage': 'train', 'quality_pass': not failures, 'failures': failures,
        'visual_approval': False, 'counterfactual_only': True,
        'train_frames': train, 'validation_frames': val, 'validation_used_for_optimization': False,
        'updates': updates, 'pos_weight': posweight, 'fixed_steps': {'A': 600, 'B': 300},
        'training_log': log, 'frames': evaluation, 'checkpoint_reload_exact': True,
        'base_not_loaded_or_optimized': True, 'train_rms_sha256': array_hash(scale),
        'channel_scale_sha256': array_hash(head.channel_scale.detach().cpu().numpy()),
        'cuda_timeline_seconds': gpu_start.elapsed_time(gpu_end) / 1000.,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}


def load_base(workspace, report, torch):
    import yaml
    from src.modules.spade_generator import SPADEDecoder
    from src.modules.warping_network import WarpingNetwork
    params = yaml.safe_load(inside(report['cfg']['models_config'], workspace).read_text())['model_params']
    models = {'warping_module': WarpingNetwork(**params['warping_module_params']).cuda(),
              'spade_generator': SPADEDecoder(**params['spade_generator_params']).cuda()}
    for name, key in (('warping_module', 'W'), ('spade_generator', 'G')):
        path = inside(report['cfg']['checkpoint_' + key], workspace)
        models[name].load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
        models[name].eval().requires_grad_(False)
    return models


def states(models):
    return {name: {kind: {n: array_hash(t.detach().cpu().numpy()) for n, t in values}
                  for kind, values in (('parameters', model.named_parameters()), ('buffers', model.named_buffers()))}
            for name, model in models.items()}


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4', *map(str, args)],
                   check=True, timeout=120)


def render_stage(args, workspace, snapshot, base, output, report, arrays, meta, manifest):
    torch = runtime(report)
    import cv2
    from PIL import Image
    from src.modules.contact_boundary_head import ContactBoundaryHead, features_from_hidden
    from src.utils.crop import prepare_paste_back, paste_back
    from src.config.inference_config import InferenceConfig
    head, provenance = head_from_checkpoint(Path(args.checkpoint), workspace, report, meta, manifest)
    require(provenance['samples_report_sha256'] == sha256(Path(args.samples) / 'report.json') and
            provenance.get('checkpoint_reload_exact') is True, 'checkpoint samples report/reload evidence mismatch')
    models = load_base(workspace, report, torch)
    before = states(models)
    require(before == {name: report['base_before'][name] for name in models}, 'base differs from snapshot')
    w, g = models['warping_module'], models['spade_generator']
    crop = bounded_npz(snapshot / 'source_crop.npz', 16, set(report['crop_arrays']))
    require('source_canvas.png' in report['files'] and 'A0_full.mp4' in report['files'], 'missing snapshot canvas/video')
    canvas = np.asarray(Image.open(snapshot / 'source_canvas.png').convert('RGB'))
    mask = prepare_paste_back(InferenceConfig().mask_crop, crop['M_c2o'], dsize=(canvas.shape[1], canvas.shape[0]))
    driving = inside(report['ArgumentConfig']['driving'], workspace)
    require(str(driving) in report['inputs_before'], 'driver not in snapshot input manifest')
    audio_before = audio_signature(media(driving, audio=True))
    require(audio_before == report['audio_before'], 'original audio changed')
    fsource, xs = torch.from_numpy(arrays['F']).cuda(), torch.from_numpy(arrays['x_s']).cuda()
    zero = ContactBoundaryHead(head.channel_scale).eval().requires_grad_(False)
    selected = set(prep.sample_indices())
    encoders, records = {}, []
    gates = np.zeros((581, 1, 50, 120), dtype=np.bool_)
    previous = None
    started_gpu = time.monotonic()
    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    height, width = canvas.shape[:2]
    require(width % 2 == height % 2 == 0, 'yuv420p requires even canvas')
    try:
        for name in ('mild', 'strong'):
            encoders[name] = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                '-framerate', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
                '-crf', '18', '-pix_fmt', 'yuv420p', str(output / f'{name}_silent.mp4')], stdin=subprocess.PIPE)
        with torch.no_grad():
            for frame in range(581):
                if frame % 25 == 0:
                    budget(workspace, base)
                with torch.autocast('cuda', dtype=torch.float16):
                    warped = w(fsource, kp_driving=torch.from_numpy(arrays['final_k'][frame]).cuda(), kp_source=xs)['out']
                    h = g.forward_features(warped)
                    require(h.dtype == torch.float16 and tuple(h.shape) == (1, 64, 256, 256), 'unexpected native H')
                    logits = g.conv_img(torch.nn.functional.leaky_relu(h, .2))
                require(logits.dtype == torch.float16, 'base logits must stay production FP16')
                features = features_from_hidden(h, h if previous is None else previous)
                base_tensor = torch.sigmoid(logits)
                bfloat, braw = tensor_image(base_tensor)
                require(array_hash(braw) == report['frames'][frame]['raw512'], f'baseline mismatch frame {frame}')
                bfull = paste_back(braw, crop['M_c2o'], canvas, mask)
                require(array_hash(bfull) == report['frames'][frame]['full_rgb'], f'baseline pasteback mismatch {frame}')
                if frame in (0, 225, 266, 446, 580):
                    require(torch.equal(head.apply_to_logits(logits, features, 0.), base_tensor) and
                            torch.equal(zero.apply_to_logits(logits, features, 1.), base_tensor), 'zero baseline mismatch')
                prediction = head(features)
                support = np.zeros((512, 512), dtype=np.bool_)
                gates[frame] = prediction['gate'].cpu().numpy()[0] > 0
                support[330:380, 200:320] = gates[frame, 0]
                roi = np.zeros_like(support)
                roi[330:380, 200:320] = True
                domain = cv2.warpAffine(support.astype(np.float32), crop['M_c2o'][:2],
                                       (width, height), flags=cv2.INTER_LINEAR) > 0
                row = {'frame': frame, 'raw512': array_hash(braw), 'full_rgb': array_hash(bfull),
                       'gate_positive_pixels': int(support.sum()), 'candidates': {}}
                if frame in selected:
                    Image.fromarray(braw).save(output / f'A0_raw512_f{frame:04d}.png')
                for name, strength in (('mild', .5), ('strong', 1.)):
                    candidate = head.apply_to_logits(logits, features, strength)
                    cfloat, craw = tensor_image(candidate)
                    full = paste_back(craw, crop['M_c2o'], canvas, mask)
                    audits = {'outside_predgate_float': pixel_audit(bfloat, cfloat, support),
                              'outside_predgate_uint8': pixel_audit(braw, craw, support),
                              'outside_fixedROI_float': pixel_audit(bfloat, cfloat, roi),
                              'outside_fixedROI_uint8': pixel_audit(braw, craw, roi),
                              'outside_full_propagation_uint8': pixel_audit(bfull, full, domain)}
                    require(all(v['outside_max'] == 0 for v in audits.values()), f'numerical leakage {frame}')
                    row['candidates'][name] = {**audits, 'raw512': array_hash(craw), 'full_rgb': array_hash(full)}
                    encoders[name].stdin.write(np.ascontiguousarray(full).tobytes())
                    if frame in selected:
                        Image.fromarray(craw).save(output / f'{name}_raw512_f{frame:04d}.png')
                    if frame in (226, 255, 446):
                        Image.fromarray(full).save(output / f'{name}_full_f{frame:04d}.png')
                if frame in (226, 255, 446):
                    Image.fromarray(bfull).save(output / f'A0_full_f{frame:04d}.png')
                records.append(row)
                previous = h.detach()
        for encoder in encoders.values():
            encoder.stdin.close()
            require(encoder.wait(timeout=60) == 0, 'stream encoder failed')
    finally:
        # Owned children only; the supervisor also owns the complete process group.
        for encoder in encoders.values():
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - started_gpu
    after = states(models)
    require(before == after and all(not m.training and all(not p.requires_grad for p in m.parameters())
                                   for m in models.values()), 'base mutated or unfrozen')
    np.savez_compressed(output / 'full_gate.npz', gate=gates)
    verification, audio_after = {}, {}
    for name in ('mild', 'strong'):
        ffmpeg('-i', output / f'{name}_silent.mp4', '-i', driving, '-map', '0:v:0', '-map', '1:a:0',
               '-c', 'copy', '-copyts', '-avoid_negative_ts', 'disabled', output / f'{name}_full.mp4')
    # A0 is the immutable snapshot video, not a copied/reconstructed baseline.
    baseline = snapshot / 'A0_full.mp4'
    filters = ';'.join(f'[{i}:v]scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]'
                       for i in range(3)) + ';[v0][v1][v2]hstack=inputs=3[v]'
    ffmpeg('-i', baseline, '-i', output / 'mild_full.mp4', '-i', output / 'strong_full.mp4', '-i', driving,
           '-filter_complex_threads', '1', '-filter_complex', filters, '-map', '[v]', '-map', '3:a:0',
           '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
           '-copyts', '-avoid_negative_ts', 'disabled', output / 'three_columns_full.mp4')
    paths = [baseline] + [output / name for name in ('mild_silent.mp4', 'strong_silent.mp4',
              'mild_full.mp4', 'strong_full.mp4', 'three_columns_full.mp4')]
    for path in paths:
        verification[str(path)] = check_video(media(path))
        ffmpeg('-xerror', '-i', path, '-f', 'null', '-')
        if not path.name.endswith('_silent.mp4'):
            audio_after[str(path)] = audio_signature(media(path, audio=True))
            require(audio_after[str(path)] == audio_before, 'audio packets/timestamps changed: ' + str(path))
    return {'stage': 'render', 'quality_pass': False,
        'quality_status': ('pending independent full-video visual review' if provenance['quality_pass']
                           else 'training gate failed; failure visualization only'),
        'training_failures': provenance.get('failures', []),
        'semantic_safety_all581': 'unknown; predicted gate is not an approved semantic mask',
        'training_candidate_quality_pass': provenance['quality_pass'], 'baseline_video': str(baseline),
        'column_order': ['A0/v1 this snapshot', 'mild strength 0.5', 'strong strength 1.0'],
        'burned_in_captions': False, 'nframes': 581, 'fps': 25, 'duration': 23.24,
        'frames': records, 'base_before': before, 'base_after': after,
        'zero_strength_and_zero_head_frames': [0, 225, 266, 446, 580],
        'video_verification': verification, 'audio_before': audio_before, 'audio_after': audio_after,
        'audio_exact': True, 'sampled_raw_frames': sorted(selected),
        'no_RGB_restoration': True, 'labels_read': False, 'sample_arrays_read': False,
        'gpu_loop_wall_seconds': gpu_seconds, 'cuda_timeline_seconds': gpu_start.elapsed_time(gpu_end) / 1000.,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('train', 'render'))
    for name in ('workspace', 'snapshot', 'samples', 'output', 'budget-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--labels')
    p.add_argument('--checkpoint')
    p.add_argument('--authorize-contact', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def validate_stage(args):
    require((args.stage == 'train' and bool(args.labels) and args.checkpoint is None) or
            (args.stage == 'render' and args.labels is None and bool(args.checkpoint)),
            'train requires labels/no checkpoint; render requires checkpoint and rejects labels')


def run_worker(args, workspace, snapshot, base, output):
    manifest, result = {}, {}
    started = time.monotonic()
    code = code_check()
    try:
        report, arrays, manifest = prep.load_inputs(snapshot, workspace, manifest)
        meta = sample_metadata(inside(args.samples, workspace), snapshot, report, workspace, manifest)
        # For pasteback reproducibility include transitive implementation/resource blobs.
        if args.stage == 'render':
            for rel in ('src/utils/crop.py', 'src/config/inference_config.py', 'src/utils/resources/mask_template.png'):
                require(command('git', 'rev-parse', report['code_sha'] + ':' + rel) ==
                        command('git', 'hash-object', str(ROOT / rel)), 'pasteback implementation changed')
                tracked(ROOT / rel, workspace, manifest)
        write_json(output / 'report.json', {'status': 'running', 'inputs_before': manifest,
                                           'supervisor_verified': False})
        action = train_stage if args.stage == 'train' else render_stage
        result = action(args, workspace, snapshot, base, output, report, arrays, meta, manifest)
        require(code_check() == code, 'source code changed')
        after = {name: sha256(name) for name in manifest}
        require(manifest == after, 'input/base weight/source mutation')
        result.update(status='completed', inputs_before=manifest, inputs_after=after,
                      weights_before=report['weights_before'],
                      weights_after={name: sha256(name) for name in report['weights_before']},
                      source_sha=report['code_sha'], code_sha=code,
                      snapshot_arrays=report['snapshot_arrays'], environment=report['environment'],
                      gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], cpu_affinity=sorted(os.sched_getaffinity(0)),
                      supervisor_verified=False, worker_wall_seconds=time.monotonic() - started,
                      files={p.name: sha256(p) for p in output.iterdir() if p.is_file() and
                             p.name not in ('report.json', 'supervisor.json')}, **budget(workspace, base))
        write_json(output / 'report.json', result)
    except BaseException as exc:
        after = {}
        for name in manifest:
            try:
                after[name] = sha256(name)
            except OSError as error:
                after[name] = {'error': repr(error)}
        result.update(status='failed', quality_pass=False, error=repr(exc), inputs_before=manifest,
                      inputs_after=after, supervisor_verified=False, worker_wall_seconds=time.monotonic() - started)
        write_json(output / 'report.json', result)
        raise


def main():
    started = time.monotonic()
    args = parser().parse_args()
    validate_stage(args)  # Reject render labels before any file access/import of tensor runtime.
    require(sys.platform == 'linux' and args.authorize_contact, 'authorized Linux execution only')
    workspace, snapshot, base, output = prep.check_paths(args)
    samples = inside(args.samples, workspace)
    require(samples.is_dir() and not samples.is_relative_to(output) and not output.is_relative_to(samples),
            'samples/output overlap')
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)) and inside(os.environ[key], workspace).is_dir(), 'missing isolated ' + key)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'one explicit GPU UUID required')
    code_check()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if args._worker:
        require(os.environ.get('CONTACT_RELEASE_PARENT') == str(os.getppid()), 'supervised worker required')
        run_worker(args, workspace, snapshot, base, output)
        return
    require(not output.exists(), 'output must be new; never overwrite')
    import fcntl
    base.mkdir(parents=True, exist_ok=True)
    # Persistent advisory lock; no artifact deletion, no concurrent cumulative-budget races.
    with (base / 'release.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        charge = elapsed_charge(base)
        limit = min(600., 1800. - charge)
        require(limit >= 600, 'less than one 600s stage remains; no partial-budget run')
        initial = budget(workspace, base)
        output.mkdir(parents=True, exist_ok=False)
        owned = timer = process = None
        status, error = 'failed', None
        write_json(output / 'supervisor.json', {'status': 'running', 'stage': args.stage,
            'reserved_seconds': 600, 'previous_wall_seconds': charge})
        try:
            env = dict(os.environ, CONTACT_RELEASE_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[key] = '4'
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                       cwd=ROOT, env=env, start_new_session=True)
            owned = OwnedProcessGroup(process)
            # Leave audit/owned-group cleanup margin inside the charged 600 seconds.
            worker_deadline = limit - 15
            timer = threading.Timer(max(0, worker_deadline - (time.monotonic() - started)), owned.kill)
            timer.daemon = True
            timer.start()
            while not owned.exited():
                require(time.monotonic() - started < worker_deadline, 'stage worker timeout (audit margin reserved)')
                budget(workspace, base)
                time.sleep(.5)
            owned.finish()
            require(process.returncode == 0 and time.monotonic() - started < limit, 'worker failed/timed out')
            result = read_json(output / 'report.json', 32 * 2**20)
            require(result['status'] == 'completed', 'worker report incomplete')
            result.update(supervisor_verified=True, budget_before=initial, budget_after=budget(workspace, base))
            write_json(output / 'report.json', result)
            status = 'completed'
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if owned is not None:
                owned.finish()
            path = output / 'report.json'
            result = read_json(path, 32 * 2**20) if path.exists() else {}
            if status != 'completed':
                result['inputs_after'] = {}
                for name in result.get('inputs_before', {}):
                    try:
                        result['inputs_after'][name] = sha256(name)
                    except OSError as exc:
                        result['inputs_after'][name] = {'error': repr(exc)}
            # Failure input rehashing is charged too, not hidden after the clock stops.
            wall = time.monotonic() - started
            record = {'status': status, 'stage': args.stage, 'error': error,
                'returncode': None if process is None else process.returncode,
                'wall_seconds': wall, 'previous_wall_seconds': charge, 'cumulative_wall_seconds': charge + wall,
                'hard_limit_seconds': limit, 'wall_budget_pass': wall <= 600 and charge + wall <= 1800}
            if not record['wall_budget_pass']:
                record['status'] = status = 'failed'
                error = error or 'wall budget exceeded during final audit'
                record['error'] = error
            result.update(supervisor=record, wall_seconds=wall)
            if status != 'completed':
                result.update(status='failed', supervisor_verified=False, quality_pass=False, error=error)
            write_json(output / 'supervisor.json', record)
            write_json(path, result)
        require(status == 'completed', 'supervisor rejected run')


if __name__ == '__main__':
    main()
