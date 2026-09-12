"""Bounded paired-video GT training and independent full-video native-G replay.

Importing this module is CPU/stdlib/NumPy only. CUDA dependencies are worker-local.
Predicted support is NOT a semantic approval. No validation-based model selection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.snapshot_upper_teeth import (OwnedProcessGroup, array_hash, audio_signature,
    budget, check_video, code_check, command, inside, media, require, sha256, validate_snapshot)
from scripts.prepare_real_teeth import SELECTED, OVERLAYS, ROI, validate_feature
from scripts.prepare_contact_samples import load_inputs as load_cross_inputs, sample_indices
from scripts.probe_upper_tail_capacity import bounded_npz
from scripts.run_contact_release import (elapsed_charge, read_json, write_json, tracked,
    runtime, load_base, states, ffmpeg, tensor_image, pixel_audit)

ARCH = 'real-teeth-native-decoder-v1'
THRESHOLD = .95
SOURCE_KEYS = {'source_input', 'F', 'x_s', 'final_k'}
CROP_KEYS = {'cropM', 'M_o2c', 'M_c2o', 'pt_crop106'}
CORE = ('src/config/models.yaml', 'src/modules/spade_generator.py', 'src/modules/util.py',
        'src/modules/warping_network.py', 'src/modules/dense_motion.py', 'src/utils/crop.py',
        'src/config/inference_config.py', 'src/utils/resources/mask_template.png')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('train', 'render-validation', 'render-cross'))
    for name in ('workspace', 'data', 'budget-root', 'output'):
        p.add_argument('--' + name, required=True)
    for name in ('labels', 'checkpoint', 'cross-snapshot'):
        p.add_argument('--' + name)
    p.add_argument('--authorize-real', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def validate_stage(args):
    if args.stage == 'train':
        require(bool(args.labels) and args.checkpoint is None and args.cross_snapshot is None,
                'train requires labels; rejects checkpoint/cross-snapshot')
    else:
        require(args.labels is None and bool(args.checkpoint), 'render requires checkpoint and rejects labels')
        require(bool(args.cross_snapshot) == (args.stage == 'render-cross'), 'cross snapshot stage mismatch')


def stage_seconds(previous, stage):
    # Reserve two complete render stages before training; never silently cut steps.
    require(0 <= previous <= 1800, 'invalid cumulative charge')
    remaining = 1800 - previous - (1200 if stage == 'train' else 0)
    require(remaining > 0, 'no budget left after render reservation')
    return min(600., remaining)


def source_arrays(directory, split, report):
    arrays = bounded_npz(directory / split / 'source.npz', 12, SOURCE_KEYS | CROP_KEYS)
    fixed = {k: arrays[k] for k in SOURCE_KEYS}
    require(validate_snapshot(fixed) == report['videos'][split]['source_arrays'], 'fixed source mismatch')
    require(np.array_equal(arrays['cropM'], arrays['M_o2c']), 'crop matrix alias differs')
    for key in ('M_o2c', 'M_c2o', 'pt_crop106'):
        a, meta = arrays[key], report['videos'][split]['crop_arrays'][key]
        require(list(a.shape) == meta['shape'] and str(a.dtype) == meta['dtype'] and
                np.isfinite(a).all() and array_hash(a) == meta['array_sha256'], 'crop metadata mismatch')
    for i, row in enumerate(report['videos'][split]['frames']):
        require(row['frame'] == i and row['key'] == array_hash(fixed['final_k'][i]), 'absolute K mismatch')
    return fixed


def replay_asset(name):
    """Render may hash/read sources and encoded comparison assets, never cached teachers/features."""
    p = Path(name)
    return p.name in ('source.npz', 'source_canvas.png', 'source_crop.png',
                      'BASE_aligned_full.mp4', 'GT_aligned_full.mp4')


def load_data(directory, workspace, manifest, training):
    report_path = tracked(directory / 'report.json', workspace, manifest)
    report = read_json(report_path, 32 * 2**20)
    require(report['status'] == 'completed' and report['supervisor_verified'] is True and
            report['base_before'] == report['base_after'] and report['models_before'] == report['models_after'] and
            report['inputs_before'] == report['inputs_after'], 'data not immutable/verified')
    require(report['feature_roi_xyxy'] == list(ROI) and report['selected_indices'] == list(SELECTED), 'data ROI/selection mismatch')
    require(set(report['videos']) == {'train', 'validation'}, 'need separate declared splits')
    require(len(set(report['inputs_before'].values())) == 2, 'same video cannot be both splits')
    require(re.fullmatch('[0-9a-f]{40}', report['code_sha']), 'invalid data SHA')
    for rel in CORE:
        blob = command('git', 'rev-parse', report['code_sha'] + ':' + rel)
        require(blob == command('git', 'hash-object', str(ROOT / rel)) ==
                command('git', 'rev-parse', 'HEAD:' + rel), 'original implementation changed: ' + rel)
        tracked(ROOT / rel, workspace, manifest)
    for key in ('models_before', 'inputs_before'):
        for path, expected in report[key].items():
            tracked(Path(path), workspace, manifest, expected)
    for name, row in report['files'].items():
        require(set(row) == {'bytes', 'sha256'} and type(row['bytes']) is int and row['bytes'] >= 0,
                'data inventory schema mismatch')
        path = inside(directory / name, directory)
        require(path.stat().st_size == row['bytes'], 'data file size mismatch')
        if training or replay_asset(name):
            tracked(path, directory, manifest, row['sha256'])
    for split, video in report['videos'].items():
        require(len(video['frames']) == 581 and [r['frame'] for r in video['selected']] == list(SELECTED), 'incomplete split')
        require(video['source_detection_count'] == video['source_feature_count'] == 1, 'source must be fixed')
        require(str(inside(video['input'], workspace)) in report['inputs_before'], 'unbound split input')
        required = {f'{split}/source.npz', f'{split}/BASE_aligned_full.mp4', f'{split}/GT_aligned_full.mp4'}
        for row in video['selected']:
            frame = row['frame']
            require(row['feature'] == f'feature_f{frame:04d}.npz', 'invalid feature filename')
            required.add(f'{split}/{row["feature"]}')
            require(report['files'][f'{split}/{row["feature"]}']['sha256'] == row['feature_sha256'], 'feature hash metadata differs')
            for kind in ('GT', 'BASE'):
                item = row['images'][kind]
                require(item['file'] == f'{kind}_f{frame:04d}.png', 'invalid image filename')
                rel = f'{split}/{item["file"]}'
                required.add(rel)
                require(report['files'][rel]['sha256'] == item['sha256'] and
                        item['array_sha256'] == video['frames'][frame]['rawGT' if kind == 'GT' else 'base'], 'image provenance differs')
        require(required <= set(report['files']), 'missing data assets')
    return report


def validate_labels(arrays):
    require(set(arrays) == {'allowed', 'protected', 'unknown'}, 'label keys mismatch')
    require(all(a.shape == (512, 512) and a.dtype == np.bool_ for a in arrays.values()), 'labels must be bool512')
    a, p, u = (arrays[k] for k in ('allowed', 'protected', 'unknown'))
    require(not (a & p | a & u | p & u).any() and (a | p | u).all(), 'labels must partition canvas')
    roi = np.zeros_like(a)
    x0, y0, x1, y1 = ROI
    roi[y0:y1, x0:x1] = True
    require(not (a & ~roi).any(), 'allowed outside ROI')


def load_labels(directory, data, report, manifest):
    path = tracked(directory / 'record.json', directory, manifest)
    record = read_json(path, 4 * 2**20)
    require(record['schema'] == 'real-teeth-supervision-v1' and record['purpose'] == 'real-video-paired-supervision' and
            record['raw_gt_unchanged'] is True and record['user_authorized_experiment'] is True and
            record['human_semantic_labels_approved'] is False and record['independent_review'] == 'passed' and
            record['ROI'] == list(ROI), 'labels not admitted for experimental real supervision')
    require(record['data_report_sha256'] == sha256(data / 'report.json') and
            record['data_code_sha'] == report['code_sha'], 'labels/data provenance mismatch')
    require(record.get('data_gate') is True, 'data_gate: insufficient reviewed positive supervision')
    labels = {s: {} for s in ('train', 'validation')}
    for row in record['samples']:
        split, frame = row['split'], row['frame']
        require(split in labels and type(frame) is int and frame in SELECTED and frame not in labels[split], 'invalid/duplicate label')
        require(row['file'] == f'{split}/labels_f{frame:04d}.npz', 'label filename mismatch')
        file = tracked(directory / row['file'], directory, manifest, row['sha256'])
        value = bounded_npz(file, 1, {'allowed', 'protected', 'unknown'})
        validate_labels(value)
        labels[split][frame] = value
    require(all(sorted(v) == list(SELECTED) for v in labels.values()), 'labels must cover both selected splits')
    positive_frames = {f for f, v in labels['train'].items() if v['allowed'].any()}
    require(len(positive_frames) >= 8 and sum(f+1 in positive_frames for f in positive_frames) >= 2,
            'data_gate: need eight train positives and two consecutive positive pairs')
    return labels, record


def adjacent_pairs(train_labels):
    pairs = [(f, f + 1) for f in sorted(train_labels) if f + 1 in train_labels and
             (train_labels[f]['allowed'].any() or train_labels[f + 1]['allowed'].any())]
    require(bool(pairs), 'data_gate: no positive consecutive train pair')
    return pairs


def native_feature(data, split, frame, report):
    a = bounded_npz(data / split / f'feature_f{frame:04d}.npz', 1, {'feature'})['feature']
    h = validate_feature(a)
    row = next(r for r in report['videos'][split]['selected'] if r['frame'] == frame)
    require(h == row['feature_array_sha256'], 'cached feature array mismatch')
    return a


def feature_rms(data, report):
    # One selected TRAIN ROI at a time; no validation reads and no whole-video float64 stack.
    squares, count = np.zeros(16, dtype=np.float64), 0
    for frame in SELECTED:
        a = native_feature(data, 'train', frame, report).astype(np.float64)
        squares += np.square(a).sum(axis=(0, 2, 3))
        count += a.shape[0] * a.shape[2] * a.shape[3]
    return np.maximum(np.sqrt(squares / count), 1e-4).astype(np.float32)


def batch_frames(step, positives, seed):
    require(bool(positives), 'no positive training masks')
    rng = np.random.default_rng(seed + step)
    return [int(positives[step % len(positives)])] + [int(f) for f in rng.choice(SELECTED, 3)]


def module_hash(module):
    return {k: array_hash(v.detach().cpu().numpy()) for k, v in module.state_dict().items()}


def replay(torch, models, decoder, arrays, frame, expected):
    """Production batch=1, absolute K of this split; no F/M/source redetection."""
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        warped = models['warping_module'](torch.from_numpy(arrays['F']).cuda(),
            kp_driving=torch.from_numpy(arrays['final_k'][frame]).cuda(),
            kp_source=torch.from_numpy(arrays['x_s']).cuda())['out']
    pre, seg, H, logits = decoder.extract_base(warped)
    require(array_hash(tensor_image(logits.sigmoid())[1]) == expected, f'baseline raw mismatch f{frame}')
    return pre, seg, H, logits


def real_gt_loss(torch, predictions, targets, bases, labels):
    """GT is an unchanged loss target, never a feature, gate or teacher RGB forward input."""
    def mean_error(error, mask):
        expanded = mask.expand_as(error)
        return error.abs().masked_select(expanded).sum() / expanded.sum().clamp_min(1)
    spatial, protected, gradients = [], [], []
    for pred, gt, base, masks in zip(predictions, targets, bases, labels):
        a, p = masks['allowed'], masks['protected']
        spatial.append(mean_error(pred - gt, a))
        protected.append(mean_error(pred - base, p))
        gx = (pred[..., 1:] - pred[..., :-1]) - (gt[..., 1:] - gt[..., :-1])
        gy = (pred[..., 1:, :] - pred[..., :-1, :]) - (gt[..., 1:, :] - gt[..., :-1, :])
        gradients.append((mean_error(gx, a[..., 1:] & a[..., :-1]) +
                          mean_error(gy, a[..., 1:, :] & a[..., :-1, :])) / 2)
    temporal = mean_error((predictions[1] - predictions[0]) - (targets[1] - targets[0]),
                          labels[0]['allowed'] & labels[1]['allowed'])
    components = {'gt_L1': sum(spatial) / 2, 'gt_gradient_L1': sum(gradients) / 2,
                  'protected_distill_L1': sum(protected) / 2, 'gt_temporal_L1': temporal}
    loss = components['gt_L1'] + .2 * components['gt_gradient_L1'] + .5 * components['protected_distill_L1'] + .1 * temporal
    return loss, components


def optimize(torch, loss, optimizer, scaler, parameters):
    require(torch.isfinite(loss).item(), 'nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in parameters if p.grad is not None]
    require(grads and all(torch.isfinite(g).all().item() for g in grads), 'nonfinite/missing gradients')
    norm = sum(float(g.detach().float().square().sum()) for g in grads) ** .5
    require(norm > 0, 'zero gradient')
    scaler.step(optimizer)
    scaler.update()
    return norm


def target_image(torch, data, split, frame, report):
    from PIL import Image
    raw = np.asarray(Image.open(data / split / f'GT_f{frame:04d}.png').convert('RGB'))
    require(raw.shape == (512, 512, 3) and array_hash(raw) == report['videos'][split]['frames'][frame]['rawGT'], 'GT altered')
    return torch.from_numpy(raw.copy()).cuda().permute(2, 0, 1)[None].float() / 255


def tensor_labels(torch, labels):
    return {k: torch.from_numpy(v.copy()).cuda()[None, None] for k, v in labels.items()}


def gate_counts(gate, label):
    active = gate.detach().cpu().numpy()[0, 0] > 0
    x0, y0, x1, y1 = ROI
    a, p, u = (label[k][y0:y1, x0:x1] for k in ('allowed', 'protected', 'unknown'))
    tp = int((active & a).sum())
    return {'active': int(active.sum()), 'allowed': int(a.sum()), 'true_positive': tp,
            'recall': tp / max(1, int(a.sum())), 'protected_active': int((active & p).sum()),
            'unknown_active': int((active & u).sum())}


def composed_audit(base_logits, candidate, gate):
    bf, br = tensor_image(base_logits.sigmoid())
    cf, cr = tensor_image(candidate)
    support = np.zeros((512, 512), dtype=np.bool_)
    x0, y0, x1, y1 = ROI
    support[y0:y1, x0:x1] = gate.detach().cpu().numpy()[0, 0] > 0
    roi = np.zeros_like(support)
    roi[y0:y1, x0:x1] = True
    audits = {name: pixel_audit(b, c, s) for name, b, c, s in (
        ('predgate_float', bf, cf, support), ('predgate_uint8', br, cr, support),
        ('ROI_float', bf, cf, roi), ('ROI_uint8', br, cr, roi))}
    require(all(v['outside_max'] == 0 for v in audits.values()), 'numerical support leakage')
    return cr, support, audits


def checkpoint_metadata(report, data, labels_record, code, blobs):
    return {'arch': ARCH, 'code_sha': code, 'blobs': blobs, 'data_code_sha': report['code_sha'],
            'data_report_sha256': sha256(data / 'report.json'), 'data_files': report['files'],
            'weights': report['models_before'], 'source_arrays': {s: v['source_arrays'] for s, v in report['videos'].items()},
            'labels': labels_record, 'ROI': list(ROI), 'threshold': THRESHOLD,
            'mask_steps': 600, 'student_steps': 300, 'mask_lr': .003, 'student_lr': 1e-5,
            'train_frames': list(SELECTED), 'validation_frames': list(SELECTED),
            'selection': 'fixed final step300; validation never selects weights/threshold', 'quality_pass': False}


def load_checkpoint(path, workspace, data, report, manifest, decoder, torch, blobs):
    path = tracked(path, workspace, manifest)
    require(path.stat().st_size <= 128 * 2**20, 'incremental checkpoint too large')
    owner = read_json(tracked(path.parent / 'report.json', workspace, manifest), 32 * 2**20)
    supervisor = read_json(tracked(path.parent / 'supervisor.json', workspace, manifest))
    require(owner['status'] == supervisor['status'] == 'completed' and owner['stage'] == supervisor['stage'] == 'train' and
            owner['supervisor_verified'] is True and owner['base_before'] == owner['base_after'] and
            owner['files'][path.name]['sha256'] == sha256(path), 'checkpoint owner training was not accepted')
    metadata_path = tracked(path.with_suffix('.json'), workspace, manifest,
                            owner['files'][path.with_suffix('.json').name]['sha256'])
    meta = read_json(metadata_path, 16 * 2**20)
    require(meta['arch'] == ARCH and meta['checkpoint_sha256'] == sha256(path) and
            meta['code_sha'] == code_check() and meta['blobs'] == blobs and
            meta['data_report_sha256'] == sha256(data / 'report.json') and meta['data_files'] == report['files'] and
            meta['weights'] == report['models_before'] and meta['data_code_sha'] == report['code_sha'] and
            meta['source_arrays'] == {s: v['source_arrays'] for s, v in report['videos'].items()} and
            meta['threshold'] == THRESHOLD and meta['ROI'] == list(ROI) and
            meta['mask_steps'] == 600 and meta['student_steps'] == 300 and meta['checkpoint_reload_exact'] is True and
            meta['train_frames'] == meta['validation_frames'] == list(SELECTED), 'checkpoint provenance mismatch')
    delta = torch.load(path, map_location='cuda', weights_only=True)
    decoder.load_delta_state_dict(delta)
    decoder.set_training_stage('eval').eval()
    return meta


def train_stage(args, workspace, data, base, output, report, manifest, models, torch, blobs):
    from src.modules.real_teeth_decoder import RealTeethDecoder
    labels, record = load_labels(inside(args.labels, workspace), data, report, manifest)
    pairs = adjacent_pairs(labels['train'])
    scale = feature_rms(data, report)
    arrays = source_arrays(data, 'train', report)
    decoder = RealTeethDecoder(models['spade_generator'], scale, THRESHOLD).cuda()
    initial_student = {n: module_hash(getattr(decoder, n)) for n in ('student_up_1', 'student_conv')}
    # No student update precedes the production FP16 copy-equivalence check.
    exact = []
    with torch.no_grad():
        for frame in OVERLAYS:
            pre, seg, H, b = replay(torch, models, decoder, arrays, frame, report['videos']['train']['frames'][frame]['base'])
            s = decoder.student_logits(pre, seg)
            require(torch.equal(s, b) and torch.equal(s.sigmoid(), b.sigmoid()) and
                    np.array_equal(tensor_image(s.sigmoid())[1], tensor_image(b.sigmoid())[1]), 'pretrained student step0 differs')
            exact.append(frame)
    decoder.set_training_stage('mask').train()
    parameters = list(decoder.mask_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.003)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    require(scaler.is_enabled(), 'GradScaler must be enabled')
    positives = [f for f in SELECTED if labels['train'][f]['allowed'].any()]
    x0, y0, x1, y1 = ROI
    pos = sum(int(v['allowed'][y0:y1, x0:x1].sum()) for v in labels['train'].values())
    neg = sum(int(v['protected'][y0:y1, x0:x1].sum()) for v in labels['train'].values())
    require(pos > 0 and neg > 0, 'data_gate: need positive and protected mask pixels')
    pos_weight = torch.tensor(min(20., neg / pos), device='cuda')
    mask_before = module_hash(decoder.mask_net)
    curve = []
    for step in range(600):
        if step % 25 == 0:
            budget(workspace, base)
        batch = batch_frames(step, positives, report['environment']['seed'])
        features = torch.from_numpy(np.concatenate([native_feature(data, 'train', f, report) for f in batch])).cuda().float()
        features = features / decoder.feature_scale
        target = torch.from_numpy(np.stack([labels['train'][f]['allowed'][y0:y1, x0:x1] for f in batch])).cuda()[:, None]
        known = torch.from_numpy(np.stack([(labels['train'][f]['allowed'] | labels['train'][f]['protected'])[y0:y1, x0:x1]
                                          for f in batch])).cuda()[:, None]
        logits, _ = decoder.predict_mask(features)
        loss_map = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight, reduction='none')
        loss = loss_map.masked_select(known).mean()
        norm = optimize(torch, loss, optimizer, scaler, parameters)
        if step in (0, 99, 299, 599):
            curve.append({'stage': 'mask', 'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': norm})
    require(module_hash(decoder.mask_net) != mask_before, 'mask parameters never updated')
    mask_frozen = module_hash(decoder.mask_net)
    decoder.set_training_stage('student').train()
    parameters = list(decoder.student_up_1.parameters()) + list(decoder.student_conv.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-5)
    for step in range(300):
        if step % 10 == 0:
            budget(workspace, base)
        pair = pairs[step % len(pairs)]
        predictions, targets, baselines, masks = [], [], [], []
        for frame in pair:  # two batch=1 forwards: production AMP does not change with batch size
            pre, seg, H, b = replay(torch, models, decoder, arrays, frame, report['videos']['train']['frames'][frame]['base'])
            predictions.append(decoder.student_logits(pre, seg).sigmoid().float())
            targets.append(target_image(torch, data, 'train', frame, report))
            baselines.append(b.sigmoid().float())
            masks.append(tensor_labels(torch, labels['train'][frame]))
        loss, components = real_gt_loss(torch, predictions, targets, baselines, masks)
        norm = optimize(torch, loss, optimizer, scaler, parameters)
        if step in (0, 99, 299):
            curve.append({'stage': 'student', 'step': step + 1, 'pair': pair, 'loss': float(loss.detach()),
                          'gradient_norm': norm, **{k: float(v.detach()) for k, v in components.items()}})
    del optimizer, predictions, targets, baselines, masks, loss, components
    require(module_hash(decoder.mask_net) == mask_frozen, 'frozen mask changed')
    final_student = {n: module_hash(getattr(decoder, n)) for n in initial_student}
    require(all(final_student[n] != initial_student[n] for n in initial_student), 'both student copies must update')
    decoder.set_training_stage('eval').eval()
    checkpoint = output / 'decoder.pt'
    torch.save(decoder.delta_state_dict(), checkpoint)
    require(checkpoint.stat().st_size <= 128 * 2**20, 'checkpoint exceeds limit')
    # Evaluation starts only after final fixed checkpoint; neither split changes any decision.
    evaluation, references = {}, {}
    with torch.no_grad():
        for split in ('train', 'validation'):
            fixed = source_arrays(data, split, report)
            rows, detailed = [], []
            for frame in SELECTED:
                features = torch.from_numpy(native_feature(data, split, frame, report)).cuda().float() / decoder.feature_scale
                _, gate = decoder.predict_mask(features)
                rows.append({'frame': frame, **gate_counts(gate, labels[split][frame])})
            for frame in OVERLAYS:
                pre, seg, H, b = replay(torch, models, decoder, fixed, frame, report['videos'][split]['frames'][frame]['base'])
                _, gate = decoder.predict_mask(decoder.mask_features(H))
                s = decoder.student_logits(pre, seg)
                out = decoder.compose(b, s, gate)
                raw, _, audits = composed_audit(b, out, gate)
                gt = target_image(torch, data, split, frame, report)
                label = tensor_labels(torch, labels[split][frame])
                def metrics(value):
                    error = (value.float() - gt).abs()
                    return {k + '_MAE': float(error.masked_select(label[k].expand_as(error)).sum() /
                            label[k].expand_as(error).sum().clamp_min(1)) for k in label}
                bf, br = tensor_image(b.sigmoid())
                cf, _ = tensor_image(out)
                _, sr = tensor_image(s.sigmoid())
                protection = {k: {'gated_uint8_changed': int(np.any(raw != br, axis=2)[labels[split][frame][k]].sum()),
                    'gated_float_max': float(np.abs(cf - bf)[labels[split][frame][k]].max(initial=0)),
                    'raw_student_uint8_changed': int(np.any(sr != br, axis=2)[labels[split][frame][k]].sum())}
                    for k in ('protected', 'unknown')}
                cached = next(r for r in rows if r['frame'] == frame)
                detailed.append({'frame': frame, 'base_vs_realGT': metrics(b.sigmoid()), 'raw_student_vs_realGT': metrics(s.sigmoid()),
                                 'gated_vs_realGT': metrics(out), 'semantic_region_changes': protection,
                                 'cached_gate_counts': cached, 'audits': audits, **gate_counts(gate, labels[split][frame])})
                if frame in (0, 255, 266):
                    references[(split, frame)] = (out.cpu().clone(), raw.copy())
            evaluation[split] = {'mask_all_selected': rows, 'replay_selected': detailed}
        # Strict weights-only reload into the same architecture, no validation-selected replacement.
        decoder.load_delta_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        for (split, frame), (tensor, raw) in references.items():
            fixed = source_arrays(data, split, report)
            pre, seg, H, b = replay(torch, models, decoder, fixed, frame, report['videos'][split]['frames'][frame]['base'])
            _, gate = decoder.predict_mask(decoder.mask_features(H))
            out = decoder.compose(b, decoder.student_logits(pre, seg), gate)
            require(torch.equal(tensor, out.cpu()) and np.array_equal(raw, tensor_image(out)[1]), 'checkpoint reload differs')
    require(module_hash(decoder.mask_net) == mask_frozen and
            {n: module_hash(getattr(decoder, n)) for n in initial_student} == final_student, 'evaluation mutated trained state')
    meta = checkpoint_metadata(report, data, record, code_check(), blobs)
    meta.update(checkpoint_sha256=sha256(checkpoint), checkpoint_reload_exact=True,
                initial_student=initial_student, final_student=final_student, frozen_mask=mask_frozen,
                feature_rms=scale.tolist(), feature_rms_split='train', mask_positive_weight=float(pos_weight),
                step0_exact_frames=exact, learning_curve=curve, evaluation=evaluation,
                semantic_gate_pass=all(r['protected_active'] == r['unknown_active'] == 0 for e in evaluation.values()
                    for r in e['mask_all_selected'] + e['replay_selected']) and
                    all(sum(r['true_positive'] for r in e['mask_all_selected']) > 0 and
                        sum(r['true_positive'] for r in e['replay_selected']) > 0 for e in evaluation.values()),
                quality_status='pending full video visual review; same-video real targets are not cross-person GT')
    write_json(checkpoint.with_suffix('.json'), meta)
    return {'checkpoint': checkpoint.name, **meta}


def render_stage(args, workspace, data, base, output, report, manifest, models, torch, blobs):
    from src.modules.real_teeth_decoder import RealTeethDecoder
    from PIL import Image
    decoder = RealTeethDecoder(models['spade_generator'], threshold=THRESHOLD).cuda()
    meta = load_checkpoint(inside(args.checkpoint, workspace), workspace, data, report, manifest, decoder, torch, blobs)
    decoder_before = {k: array_hash(v.cpu().numpy()) for k, v in decoder.delta_state_dict().items()}
    cross = args.stage == 'render-cross'
    if cross:
        import cv2
        from src.config.inference_config import InferenceConfig
        from src.utils.crop import paste_back, prepare_paste_back
        cv2.setNumThreads(4)
        snapshot = inside(args.cross_snapshot, workspace)
        snap, arrays, _ = load_cross_inputs(snapshot, workspace, manifest)
        require(snap['code_sha'] == '8465cf4a619dd8381c7be1f421de31c4e6ecd27c', 'expected fixed 8465 HD snapshot')
        require(snap['base_before'] == report['base_before'] and
                all(report['models_before'].get(k) == v for k, v in snap['weights_before'].items()), 'cross base/weights differ')
        require(snap['environment']['torch'] == report['environment']['torch'] and
                snap['environment']['cuda'] == report['environment']['cuda'] and snap['environment']['tf32'] is False and
                snap['cfg']['flag_use_half_precision'] is True, 'cross runtime incompatible')
        crop = bounded_npz(snapshot / 'source_crop.npz', 16, set(snap['crop_arrays']))
        canvas = np.asarray(Image.open(snapshot / 'source_canvas.png').convert('RGB'))
        height, width = canvas.shape[:2]
        mask_crop = InferenceConfig().mask_crop
        require(array_hash(mask_crop) == snap['cfg']['mask_crop']['array_sha256'], 'pasteback template changed')
        mask = prepare_paste_back(mask_crop, crop['M_c2o'], dsize=(width, height))
        driving = inside(snap['ArgumentConfig']['driving'], workspace)
        comparisons = [snapshot / 'A0_full.mp4']
        records = snap['frames']
        expected_audio = snap['audio_before']
    else:
        arrays = source_arrays(data, 'validation', report)
        video = report['videos']['validation']
        records, expected_audio = video['frames'], video['audio_before']
        driving = inside(video['input'], workspace)
        height = width = 512
        # Encoded GT is only an ffmpeg comparison input after all model outputs have been produced.
        comparisons = [data / 'validation/GT_aligned_full.mp4', data / 'validation/BASE_aligned_full.mp4']
    require(width % 2 == height % 2 == 0, 'even output canvas required')
    require(audio_signature(media(driving, audio=True)) == expected_audio, 'source audio differs')
    encoders, rows = {}, []
    gates = np.zeros((581, 1, 120, 190), dtype=np.bool_)
    selected = set(sample_indices())
    try:
        for name in ('mild', 'strong'):
            encoders[name] = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '25',
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p',
                str(output / f'{name}_silent.mp4')], stdin=subprocess.PIPE)
        with torch.no_grad():
            for frame in range(581):
                if frame % 10 == 0:
                    budget(workspace, base)
                expected = records[frame]['raw512' if cross else 'base']
                pre, seg, H, b = replay(torch, models, decoder, arrays, frame, expected)
                s = decoder.student_logits(pre, seg)
                _, gate = decoder.predict_mask(decoder.mask_features(H))
                require(torch.equal(decoder.compose(b, s, gate, 0), b.sigmoid()), 'strength0 differs')
                gates[frame] = gate.cpu().numpy()[0] > 0
                br = tensor_image(b.sigmoid())[1]
                bfull = paste_back(br, crop['M_c2o'], canvas, mask) if cross else br
                if cross:
                    require(array_hash(bfull) == records[frame]['full_rgb'], 'cross baseline full hash differs')
                row = {'frame': frame, 'base_raw': array_hash(br), 'base_full': array_hash(bfull),
                       'gate_pixels': int(gates[frame].sum()), 'candidates': {}}
                panels = [br] if frame in selected else None
                for name, strength in (('mild', .5), ('strong', 1.)):
                    candidate = decoder.compose(b, s, gate, strength)
                    raw, support, audits = composed_audit(b, candidate, gate)
                    full = paste_back(raw, crop['M_c2o'], canvas, mask) if cross else raw
                    if cross:
                        domain = cv2.warpAffine(support.astype(np.float32), crop['M_c2o'][:2],
                                                (width, height), flags=cv2.INTER_LINEAR) > 0
                        audits['pasteback_propagation_uint8'] = pixel_audit(bfull, full, domain)
                        require(audits['pasteback_propagation_uint8']['outside_max'] == 0, 'pasteback leakage')
                    encoders[name].stdin.write(np.ascontiguousarray(full).tobytes())
                    row['candidates'][name] = {'raw': array_hash(raw), 'full': array_hash(full), 'audits': audits}
                    if panels is not None:
                        panels.append(raw)
                if panels is not None:
                    Image.fromarray(np.concatenate(panels, axis=1)).save(output / f'raw_three_f{frame:04d}.png')
                rows.append(row)
        for proc in encoders.values():
            proc.stdin.close()
            require(proc.wait(timeout=60) == 0, 'stream encoder failed')
    finally:
        for proc in encoders.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    require(decoder_before == {k: array_hash(v.cpu().numpy()) for k, v in decoder.delta_state_dict().items()}, 'render mutated decoder')
    np.savez_compressed(output / 'full_gate.npz', gate=gates)
    for name in ('mild', 'strong'):
        ffmpeg('-i', output / f'{name}_silent.mp4', '-i', driving, '-map', '0:v:0', '-map', '1:a:0',
               '-c', 'copy', '-copyts', '-avoid_negative_ts', 'disabled', output / f'{name}_full.mp4')
    inputs = comparisons + [output / 'mild_full.mp4', output / 'strong_full.mp4']
    count = len(inputs)
    filters = ';'.join(f'[{i}:v]scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]'
                       for i in range(count)) + ';' + ''.join(f'[v{i}]' for i in range(count)) + f'hstack=inputs={count}[v]'
    columns = output / ('three_columns_full.mp4' if cross else 'four_columns_full.mp4')
    ffmpeg(*[arg for path in inputs + [driving] for arg in ('-i', path)], '-filter_complex_threads', '1',
           '-filter_complex', filters, '-map', '[v]', '-map', f'{count}:a:0', '-c:v', 'libx264', '-threads', '4',
           '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-copyts', '-avoid_negative_ts', 'disabled', columns)
    verification = {}
    for path in inputs + [columns] + [output / f'{name}_silent.mp4' for name in ('mild', 'strong')]:
        stream = check_video(media(path))
        ffmpeg('-xerror', '-i', path, '-f', 'null', '-')
        audio_exact = None
        if not path.name.endswith('_silent.mp4'):
            audio_exact = audio_signature(media(path, audio=True)) == expected_audio
            require(audio_exact, 'audio packets/PTS/DTS differ')
        verification[str(path)] = {'stream': stream, 'full_decode_passed': True, 'audio_exact': audio_exact}
    return {'quality_pass': False, 'quality_status': 'full video pending parent visual/semantic review',
            'semantic_safety': 'unknown: prediction support is not a semantic label', 'labels_read': False,
            'GT_prediction_input': False, 'sample_features_read': False, 'checkpoint_sha256': meta['checkpoint_sha256'],
            'nframes': 581, 'fps': 25, 'duration': 23.24, 'frames': rows, 'video_verification': verification,
            'column_order': ['A0', 'mild', 'strong'] if cross else ['real aligned GT', 'BASE', 'mild', 'strong'],
            'sampled_raw_frames': sorted(selected), 'audio_exact': True}


def inventory(output):
    # Logs and supervisor/report are mutable; do not self-hash or hash a live log.
    return {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in
            ('report.json', 'supervisor.json', 'worker.log', 'failure.json') and not p.name.endswith('.writing')}


def worker(args, workspace, data, base, output):
    manifest = {}
    code = code_check()
    blobs = {rel: command('git', 'hash-object', str(ROOT / rel)) for rel in (*CORE,
        'src/modules/real_teeth_decoder.py', 'scripts/run_real_teeth.py')}
    report = load_data(data, workspace, manifest, args.stage == 'train')
    torch = runtime(report)
    models = load_base(workspace, report, torch)
    before = states(models)
    require(before == {n: report['base_before'][n] for n in models}, 'base W/G differs')
    try:
        fn = train_stage if args.stage == 'train' else render_stage
        result = fn(args, workspace, data, base, output, report, manifest, models, torch, blobs)
        after = states(models)
        require(before == after and all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in models.values()), 'base mutated/unfrozen')
        require(code_check() == code, 'code changed')
        require(all(sha256(p) == h for p, h in manifest.items()), 'input/model file changed')
        result.update(status='completed', stage=args.stage, code_sha=code, blobs=blobs,
                      base_before=before, base_after=after, input_hashes=manifest, files=inventory(output),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'report.json', result)
    except BaseException as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'stage': args.stage, 'error': repr(exc),
                   'input_hashes': manifest, 'base_before': before, 'base_after': states(models)})
        raise


def main():
    started = time.monotonic()
    args = parser().parse_args()
    validate_stage(args)
    require(sys.platform == 'linux' and args.authorize_real, 'Linux and --authorize-real required')
    workspace = Path(args.workspace).resolve(strict=True)
    inside(ROOT, workspace)
    data, base = inside(args.data, workspace), inside(args.budget_root, workspace)
    output = inside(args.output, base, exists=False)
    require(data.is_dir() and base.is_dir() and output != base and
            not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'invalid data/budget/output roots')
    require(not output.is_relative_to(data) and not data.is_relative_to(output), 'data/output must be distinct')
    require(not Path(args.output).is_symlink() and not Path(args.budget_root).is_symlink(), 'symlink output/budget forbidden')
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)) and inside(os.environ[key], workspace).is_dir(), 'isolated cache required: ' + key)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'one GPU UUID required')
    code_check()
    if args._worker:
        require(os.environ.get('REAL_TEETH_RUN_PARENT') == str(os.getppid()), 'worker must be supervised')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
                os.environ.get('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT') == '1', 'determinism/process ownership environment required')
        cpus = sorted(os.sched_getaffinity(0))
        require(len(cpus) >= 4, 'four CPU affinity required')
        os.sched_setaffinity(0, cpus[:4])
        worker(args, workspace, data, base, output)
        return
    require(not output.exists(), 'new output required; partial runs are never overwritten')
    prior = elapsed_charge(base)
    limit = stage_seconds(prior, args.stage)
    initial_budget = budget(workspace, base)
    output.mkdir(parents=True)
    write_json(output / 'supervisor.json', {'status': 'running', 'wall_seconds': 0., 'stage': args.stage})
    env = dict(os.environ, REAL_TEETH_RUN_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               IMAGEIO_FFMPEG_NO_PREVENT_SIGINT='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    process = owned = timer = None
    completed, failure = False, None
    try:
        with open(output / 'worker.log', 'xb') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                cwd=ROOT, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        owned = OwnedProcessGroup(process)
        timer = threading.Timer(max(0., limit - (time.monotonic() - started)), owned.kill)
        timer.daemon = True
        timer.start()
        while not owned.exited():
            require(time.monotonic() - started < limit, 'stage/cumulative timeout')
            budget(workspace, base)
            time.sleep(.5)
        owned.finish()
        require(process.returncode == 0, 'worker failed; see log/failure, no candidate accepted')
        result = read_json(output / 'report.json', 32 * 2**20)
        require(result['status'] == 'completed' and time.monotonic() - started < limit, 'incomplete/late worker')
        result.update(supervisor_verified=True, budget_before=initial_budget, budget_after=budget(workspace, base),
                      prior_charge_seconds=prior, wall_seconds=time.monotonic() - started)
        write_json(output / 'report.json', result)
        completed = True
    except BaseException as exc:
        failure = {'error': repr(exc), 'wall_seconds': time.monotonic() - started}
        # Preserve worker forensic details, if present.
        file = output / 'failure.json'
        detail = read_json(file, 32 * 2**20) if file.exists() else {}
        write_json(file, {**detail, 'supervisor_failure': failure})
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if owned is not None:
            owned.finish()
        write_json(output / 'supervisor.json', {'status': 'completed' if completed else 'failed', 'stage': args.stage,
            'wall_seconds': time.monotonic() - started, 'hard_limit_seconds': limit, 'prior_charge_seconds': prior,
            'returncode': process.returncode if process else None, 'failure': failure})


if __name__ == '__main__':
    main()
