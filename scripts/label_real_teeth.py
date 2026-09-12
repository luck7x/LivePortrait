"""Unapproved NumPy/PIL semantic candidates for verified real-video pairs.

203 inner/outer topology is provisional, not human semantic approval. No GT
recoloring, target construction, model inference or training is performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.snapshot_upper_teeth import budget, inside, require, sha256

ROI = (160, 290, 350, 410)
SELECTED = tuple(sorted(set(range(200, 301)) | set(range(442, 472)) | {0, 80, 116, 348, 580}))
OVERLAYS = (0, 225, 228, 251, 255, 266, 446)
EVENTS = frozenset(range(224, 230)) | frozenset(range(253, 260))
ROI_MASK = np.zeros((512, 512), dtype=bool)
ROI_MASK[290:410, 160:350] = True


def validate(rgb, lm):
    require(isinstance(rgb, np.ndarray) and rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8,
            'RGB must be uint8 512x512x3')
    require(isinstance(lm, np.ndarray) and lm.shape == (203, 2) and lm.dtype.kind == 'f'
            and np.isfinite(lm).all() and (lm >= 0).all() and (lm < 512).all(),
            'landmarks must be finite float 203x2 inside canvas')
    require(np.linalg.norm(lm[66] - lm[48]) >= 3, 'invalid mouth width')


def polygon(points):
    image = Image.new('1', (512, 512))
    ImageDraw.Draw(image).polygon([tuple(map(float, p)) for p in points], fill=1)
    return np.asarray(image, dtype=bool).copy()


def erode(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return np.logical_and.reduce([padded[y:y+512, x:x+512] for y in range(3) for x in range(3)])


def infer_regions(rgb512, lmk203):
    """Return conservative masks and evidence; unresolved inner mouth is unknown."""
    validate(rgb512, lmk203)
    outer = polygon(lmk203[48:84])
    inner = polygon(lmk203[84:108]) & outer
    core = erode(inner)
    lip = outer & ~core  # includes one-pixel inner boundary
    upper = np.zeros_like(inner)
    lower = np.zeros_like(inner)
    u = lmk203[66] - lmk203[48]
    width = float(np.linalg.norm(u))
    u = u / width
    v = np.array([-u[1], u[0]])
    if np.dot(v, lmk203[102] - lmk203[90]) < 0:
        v = -v
    height = float(np.dot(lmk203[102] - lmk203[90], v))
    yy, xx = np.indices(inner.shape)
    relative = np.stack((xx - lmk203[90, 0], yy - lmk203[90, 1]), axis=-1)
    pu, pv = relative @ u, relative @ v
    center = float(np.dot((lmk203[48] + lmk203[66]) / 2 - lmk203[90], u))
    r, g, b = rgb512.astype(np.float32).transpose(2, 0, 1)
    neutral = (r > 90) & (g > 70) & ((r-g) < .3*r) & ((r-b) < .45*r)
    gray = (.299*r + .587*g + .114*b) / 255
    central = core & (np.abs(pu-center) <= .3*width)
    bins = np.floor(pv).astype(np.int32)
    values = np.unique(bins[central])
    profile = {int(k): float(gray[central & (bins == k)].mean()) for k in values}
    tooth_bins = {int(k) for k in values if np.any(central & (bins == k) & neutral)}
    peaks = [k for k in sorted(tooth_bins) if
             profile[k] >= profile.get(k-1, -1) and profile[k] >= profile.get(k+1, -1)]
    best = None
    for a in peaks:
        for z in peaks:
            between = [k for k in profile if a < k < z]
            if z-a < 3 or not between:
                continue
            valley = min(between, key=lambda k: profile[k])
            contrast = min(profile[a], profile[z]) - profile[valley]
            if best is None or contrast > best[0]:
                best = (contrast, valley, a, z)
    contrast = float(best[0]) if best else 0.
    status = 'unresolved-mouth'
    valley = None
    # Check even the uneroded interior, so a thin visible bright tooth cannot be
    # classified closed merely because erosion removed its support.
    if height < 1.5 and not np.any(inner & neutral):
        status = 'confirmed-closed'
        lip = outer.copy()
    elif best is not None and contrast >= 18/255:
        status = 'separated-candidate'
        valley = best[1] + .5
        upper = core & (pv < valley-1)
        lower = core & (pv > valley+1.5)
    if status == 'unresolved-mouth' and not core.any():
        # Erosion must not turn an unresolved thin bright mouth into all-negative.
        lip = outer & ~inner
    unknown = outer & ~(upper | lower | lip)
    side = {'left': int((upper & (pu < center-width*.1)).sum()),
            'center': int((upper & (np.abs(pu-center) <= width*.1)).sum()),
            'right': int((upper & (pu > center+width*.1)).sum())}
    coords = np.argwhere(inner)
    bbox = None if not len(coords) else [int(coords[:, 1].min()), int(coords[:, 0].min()),
                                        int(coords[:, 1].max()+1), int(coords[:, 0].max()+1)]
    stats = {'status': status, 'contrast': contrast, 'valley_v': valley,
             'inner_height_px': height, 'mouth_width_px': width, 'inner_bbox_xyxy': bbox,
             'inner_pixels': int(inner.sum()), 'upper_pixels': int(upper.sum()),
             'lower_pixels': int(lower.sum()), 'unknown_pixels': int(unknown.sum()),
             'side_support': side, 'index_semantics_human_approved': False}
    return dict(upper=upper, lower=lower, lip=lip, inner=inner, outer=outer, unknown=unknown, stats=stats)


def _pair(gt, base, lgt, lbase):
    a, b = infer_regions(gt, lgt), infer_regions(base, lbase)
    distances = [float(np.linalg.norm(lgt[idx].mean(0)-lbase[idx].mean(0)))
                 for idx in ([0, 6, 12, 18], [24, 30, 36, 42])]
    rejected = max(distances) > 6
    mouth = a['outer'] | b['outer']
    protected = ~mouth
    allowed = np.zeros_like(mouth)
    if not rejected:
        protected |= a['lip'] | b['lip'] | a['lower'] | b['lower']
        allowed = a['upper'] & b['upper'] & ~protected & ROI_MASK
    unknown = ~(allowed | protected)
    yy, xx = np.indices(allowed.shape)
    axis = lgt[66] - lgt[48]
    width = float(np.linalg.norm(axis))
    center = (lgt[48] + lgt[66]) / 2
    horizontal = ((xx-center[0])*axis[0] + (yy-center[1])*axis[1]) / width
    side_support = {'left': int((allowed & (horizontal < -.1*width)).sum()),
                    'center': int((allowed & (np.abs(horizontal) <= .1*width)).sum()),
                    'right': int((allowed & (horizontal > .1*width)).sum())}
    stats = {'GT': a['stats'], 'BASE': b['stats'], 'alignment_rejected': rejected,
             'actual_allowed_side_support': side_support,
             'eye_center_distances_px': distances, 'alignment_threshold_px': 6,
             'alignment_threshold_kind': 'preregistered engineering screen; not proof of matching',
             'allowed_pixels': int(allowed.sum()), 'protected_pixels': int(protected.sum()),
             'unknown_pixels': int(unknown.sum())}
    return (allowed, protected, unknown), stats


def pair_regions(GT, BASE, lmkgt, lmkbase):
    """Return exactly three disjoint boolean 512 masks: allowed/protected/unknown."""
    return _pair(GT, BASE, lmkgt, lmkbase)[0]


def gate_counts(samples):
    counts = {}
    for split in ('train', 'validation'):
        positive = {s['frame'] for s in samples if s['split'] == split and s['stats']['allowed_pixels'] > 0}
        events = positive & EVENTS
        rows = [s for s in samples if s['split'] == split]
        counts[split] = {'selected_frames': len(rows),
                         'alignment_rejected_frames': sum(s['stats'].get('alignment_rejected', False) for s in rows),
                         'positive_frames': len(positive), 'event_positive_frames': len(events),
                         'event_adjacent_positive_pairs': sum(f+1 in events for f in events),
                         'all_adjacent_positive_pairs': sum(f+1 in positive for f in positive)}
    train = counts['train']
    # Eight positive frames was the overall data floor, not eight frames in a
    # thirteen-frame event list containing closed/occluded frames by design.
    return counts, train['positive_frames'] >= 8 and train['all_adjacent_positive_pairs'] >= 2


def verify_contract(data):
    report_path = data / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    require(report.get('status') == 'completed' and report.get('supervisor_verified') is True,
            'data report must be completed and supervisor verified')
    require(report.get('selected_indices') == list(SELECTED) and
            report.get('feature_roi_xyxy') == list(ROI), 'unexpected selected/ROI contract')
    require(isinstance(report.get('files'), dict) and isinstance(report.get('code_sha'), str), 'invalid inventory/code')
    truth = {}
    for split in ('train', 'validation'):
        names = ['annotations.npz', 'source.npz', 'source_canvas.png', 'source_crop.png']
        names += [f'{kind}_f{f:04d}.png' for f in SELECTED for kind in ('GT', 'BASE')]
        for name in names:
            rel = f'{split}/{name}'
            item = report['files'].get(rel)
            require(isinstance(item, dict), 'missing inventory: ' + rel)
            path = inside(data / rel, data)
            digest = sha256(path)
            require(path.stat().st_size == item['bytes'] and digest == item['sha256'], 'inventory mismatch: ' + rel)
            truth[rel] = digest
    return report, sha256(report_path), truth


def overview(gt, base, masks, frame, stats):
    x0, y0, x1, y1 = ROI
    panels = [gt.copy(), base.copy()]
    for rgb in (gt, base):
        overlay = rgb.copy()
        for mask, color in ((masks[0], (0, 100, 255)), (masks[2], (255, 220, 0))):
            overlay[mask] = np.rint(.55*rgb[mask] + .45*np.asarray(color)).astype(np.uint8)
        panels.append(overlay)
    w, h = (x1-x0)*2, (y1-y0)*2
    out = Image.new('RGB', (w*4, h+50), 'black')
    draw = ImageDraw.Draw(out)
    for i, (panel, title) in enumerate(zip(panels, ('GT unchanged', 'BASE unchanged', 'GT pair labels', 'BASE pair labels'))):
        out.paste(Image.fromarray(panel[y0:y1, x0:x1]).resize((w, h), Image.Resampling.NEAREST), (i*w, 50))
        draw.text((i*w+4, 4), f'{title} f{frame}', fill='white')
    draw.text((4, 22), f'UNAPPROVED blue=allowed yellow=unknown; positive pixels={stats["allowed_pixels"]}; '
              f'GT {stats["GT"]["status"]}; BASE {stats["BASE"]["status"]}', fill='yellow')
    return out


def worker(workspace, data, output, base):
    started = time.monotonic()
    report, report_hash, truth = verify_contract(data)
    samples = []
    for split in ('train', 'validation'):
        (output / split).mkdir()
        with np.load(data / split / 'annotations.npz', allow_pickle=False) as annotations:
            require(set(annotations.files) == {'selected', 'lmks_gt', 'lmks_base'}, 'unexpected annotations keys')
            require(np.array_equal(annotations['selected'], SELECTED), 'annotations selected mismatch')
            lgt, lbase = annotations['lmks_gt'], annotations['lmks_base']
            require(lgt.shape == lbase.shape == (136, 203, 2), 'annotation shape mismatch')
        for index, frame in enumerate(SELECTED):
            if index % 16 == 0:
                budget(workspace, base)
            images = []
            for kind in ('GT', 'BASE'):
                with Image.open(data / split / f'{kind}_f{frame:04d}.png') as im:
                    require(im.mode == 'RGB' and im.size == (512, 512), 'invalid original PNG')
                    images.append(np.array(im))
            masks, stats = _pair(*images, lgt[index], lbase[index])
            path = output / split / f'labels_f{frame:04d}.npz'
            np.savez_compressed(path, **dict(zip(('allowed', 'protected', 'unknown'), masks)))
            samples.append({'split': split, 'frame': frame, 'file': path.relative_to(output).as_posix(),
                            'sha256': sha256(path), 'stats': stats})
            if frame in OVERLAYS:
                overview(*images, masks, frame, stats).save(output / split / f'overview_f{frame:04d}.png')
    # Recheck truth after processing: original PNGs/annotations/source are read-only.
    _, after_hash, after_truth = verify_contract(data)
    require(after_hash == report_hash and after_truth == truth, 'input changed during labeling')
    counts, passed = gate_counts(samples)
    record = {'schema': 'real-teeth-supervision-v1', 'purpose': 'real-video-paired-supervision',
              'status': 'completed', 'raw_gt_unchanged': True, 'user_authorized_experiment': True,
              'human_semantic_labels_approved': False, 'independent_review': 'pending',
              'data_report_sha256': report_hash, 'data_code_sha': report['code_sha'],
              'label_code_sha256': sha256(Path(__file__)), 'ROI': list(ROI), 'samples': samples,
              'all_truth_files_sha256': truth, 'valid_frame_counts': counts, 'data_gate': passed,
              'event_frames': sorted(EVENTS), 'gate_thresholds': {'train_positive_frames': 8,
                  'train_adjacent_positive_pairs': 2},
              'gate_revision': 'Restore originally requested overall positive-frame floor; event coverage remains a separate disclosed statistic. No mask pixels or unknown labels changed.',
              'notes': 'Candidate topology and gate truth require independent visual review. No human approval; '
                       'unresolved mouth is unknown, not a negative. Numeric gate is not training approval.',
              'worker_wall_seconds': time.monotonic()-started, 'budget_after': budget(workspace, base)}
    (output / 'record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'data', 'output', 'budget-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--authorize-labels', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    require(sys.platform == 'linux' and args.authorize_labels, 'Linux and --authorize-labels required')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    workspace = Path(args.workspace).resolve(strict=True)
    base = inside(args.budget_root, workspace)
    data = inside(args.data, base)
    output = inside(args.output, base, exists=False)
    require(data.is_dir() and data.parent == base and output.parent == base and output != data,
            'data/output must be different direct siblings in budget-root')
    require(not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'budget-root separate from code')
    require(not Path(args.output).is_symlink(), 'output symlink forbidden')
    if args._worker:
        require(os.environ.get('LABEL_PARENT') == str(os.getppid()), 'supervised worker required')
        worker(workspace, data, output, base)
        return
    require(not output.exists(), 'refuse existing output; no deletion/overwrite')
    initial = budget(workspace, base)
    output.mkdir()
    proc = timer = None
    completed = False
    failure = None
    try:
        env = dict(os.environ, LABEL_PARENT=str(os.getpid()), CUDA_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with open(output / 'worker.log', 'xb') as log:
            proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        def stop_owned_worker():
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        timer = threading.Timer(max(0, 300-(time.monotonic()-started)), stop_owned_worker)
        timer.daemon = True
        timer.start()
        while proc.poll() is None:
            require(time.monotonic()-started < 300, '300s CPU wall-clock limit')
            budget(workspace, base)
            time.sleep(.2)
        require(proc.returncode == 0 and time.monotonic()-started < 300, 'worker failed/timeout')
        record_path = output / 'record.json'
        record = json.loads(record_path.read_text(encoding='utf-8'))
        require(record['status'] == 'completed', 'incomplete record')
        record.update(supervisor_verified=True, wall_seconds=time.monotonic()-started,
                      budget_before=initial, budget_after=budget(workspace, base))
        record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        completed = True
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        summary = {'status': 'completed' if completed else 'failed', 'failure': failure,
                   'wall_seconds': time.monotonic()-started, 'hard_limit_seconds': 300,
                   'returncode': proc.returncode if proc is not None else None}
        (output / 'supervisor.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
