"""Authorized Linux-only diagnostic video runner. OpenCV is imported only in main."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upper_teeth.gap_cleanup import cleanup, transition_weights

LIMIT = 20 * 1024**3


def track(cv2, reference, current, roi):
    """Return original-scale anchor->current partial affine or reject."""
    h, w = reference.shape[:2]
    factor = min(1.0, 640 / max(h, w))
    size = (round(w * factor), round(h * factor))
    gray = [cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_RGB2GRAY) for im in (reference, current)]
    x, y, bw, _ = roi
    mask = np.zeros(gray[0].shape, np.uint8)
    # Upper face only, no mouth or lower jaw; broad enough for brow/eye features.
    left, right = max(0, x - 0.6 * bw), min(w, x + 1.6 * bw)
    top, bottom = max(0, y - 1.8 * bw), max(0, y - bw * 0.35)
    mask[round(top * factor):round(bottom * factor), round(left * factor):round(right * factor)] = 255
    p = cv2.goodFeaturesToTrack(gray[0], 160, 0.01, 5, mask=mask)
    if p is None or len(p) < 8:
        return None, 'tracking-features'
    q, ok, _ = cv2.calcOpticalFlowPyrLK(gray[0], gray[1], p, None)
    if q is None:
        return None, 'tracking-forward'
    back, rev, _ = cv2.calcOpticalFlowPyrLK(gray[1], gray[0], q, None)
    if back is None:
        return None, 'tracking-backward'
    tolerance = 1.5 * bw / 75 * factor
    good = ok.ravel().astype(bool) & rev.ravel().astype(bool)
    good &= np.isfinite(q).all(axis=(1, 2)) & (np.linalg.norm(back - p, axis=2).ravel() <= tolerance)
    if good.sum() < 8:
        return None, 'tracking-cycle'
    matrix, inliers = cv2.estimateAffinePartial2D(p[good], q[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=tolerance)
    if matrix is None or inliers is None or inliers.sum() < 8 or inliers.mean() < 0.6:
        return None, 'tracking-inliers'
    predicted = p[good, 0] @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(predicted - q[good, 0], axis=1)
    scale = np.linalg.norm(matrix[:, 0])
    angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    if (not np.isfinite(matrix).all() or not 0.75 <= scale <= 1.25
            or abs(angle) > 20 or np.linalg.norm(matrix[:, 2]) > 100 * factor
            or errors[inliers.ravel().astype(bool)].max() > tolerance):
        return None, 'tracking-transform'
    matrix[:, 2] /= factor
    return matrix, 'ok'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--authorize-experimental-preview', action='store_true')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--end', required=True, type=int)
    parser.add_argument('--anchor', required=True, type=int)
    parser.add_argument('--upper-roi', required=True, nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--strength', type=float, default=0.85)
    parser.add_argument('--ema', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or not args.authorize_experimental_preview:
        parser.error('actual processing requires Linux and explicit experimental authorization')
    root, source, output = args.workspace.resolve(), args.input.resolve(), args.output.resolve()
    if not root.is_dir() or not source.is_file() or not source.is_relative_to(root) or not output.is_relative_to(root):
        parser.error('resolved input/output must lie in existing workspace')
    if output.exists() or not output.parent.is_dir():
        parser.error('output must be new and its parent must already exist')
    if not 0 <= args.start <= args.anchor <= args.end < 600 or args.end - args.start + 1 > 96:
        parser.error('invalid inclusive frame window (max 96, video max 600)')
    if not np.isfinite(args.strength) or not 0 <= args.strength <= 1:
        parser.error('invalid strength')
    if shutil.which('ffmpeg') is None:
        parser.error('existing ffmpeg required')
    import cv2
    cv2.setNumThreads(2)
    deadline = time.monotonic() + 300
    repo = Path(__file__).resolve().parents[1]
    files = ['upper_teeth/__init__.py', 'upper_teeth/gap_cleanup.py', 'scripts/probe_upper_gap.py']
    initial_sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True, timeout=10).strip():
        raise RuntimeError('require a clean pinned code worktree')
    initial_hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files}
    # Match the project's allocated-space budget; summing logical file sizes
    # double-counts environment/cache hardlinks and falsely exhausts the quota.
    initial_bytes = int(subprocess.check_output(['du', '-s', '-B1', str(root)], text=True, timeout=20).split()[0])
    def budget(reserve=0):
        if time.monotonic() >= deadline:
            raise RuntimeError('300 second deadline exceeded')
        added = sum(p.stat().st_blocks * 512 for p in output.rglob('*') if p.is_file()) if output.exists() else 0
        if initial_bytes + added + reserve > LIMIT or shutil.disk_usage(root).free < reserve + 64 * 1024**2:
            raise RuntimeError('20 GiB workspace / remaining disk headroom limit')
    budget()
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            budget()
            digest.update(block)
    cap = cv2.VideoCapture(str(source))
    frames = []
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or not 0 < fps <= 120:
        cap.release()
        raise ValueError('input FPS unavailable or unsupported; never invent timing')
    try:
        while True:
            budget()
            ok, bgr = cap.read()
            if not ok:
                break
            if count >= 600:
                raise ValueError('video exceeds 600 frames')
            h, w = bgr.shape[:2]
            if max(h, w) > 1280 or min(h, w) <= 0:
                raise ValueError('canvas exceeds 1280 dimension cache bound')
            if args.start <= count <= args.end:
                if frames and frames[0].shape != bgr.shape:
                    raise ValueError('variable canvas size')
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            count += 1
    finally:
        cap.release()
    n = args.end - args.start + 1
    if len(frames) != n:
        raise ValueError('requested real frames unavailable')
    reference = frames[args.anchor - args.start]
    h, w = reference.shape[:2]
    x, y, bw, bh = args.upper_roi
    if min(bw, bh) <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
        raise ValueError('ROI outside canvas')
    scale = bw / 75
    left, top = max(0, x - round(5 * scale)), max(0, y - round(4 * scale))
    right, bottom = min(w, x + bw + round(5 * scale)), min(h, y + bh + round(18 * scale))
    local_bbox = (x - left, y - top, bw, bh)
    # PNG worst case plus temporary video encodings, no deletion of failed artifacts.
    per_frame = h * w * 20 + 1024**2
    budget(n * per_frame)
    output.mkdir()
    for name in ('before', 'unfaded', 'after', 'allowed'):
        (output / name).mkdir()
    records, previous = [], None
    for i, before in enumerate(frames):
        budget((n - i) * per_frame)
        matrix, reason = track(cv2, reference, before, args.upper_roi)
        after = before.copy()
        allowed = np.zeros((h, w), bool)
        if matrix is not None:
            inverse = cv2.invertAffineTransform(matrix)
            aligned = cv2.warpAffine(before, inverse, (w, h), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(np.ones((h, w), np.uint8), inverse, (w, h), flags=cv2.INTER_NEAREST)
            patch = aligned[top:bottom, left:right]
            if not valid[top:bottom, left:right].all():
                previous, reason = None, 'aligned-context-outside-canvas'
            else:
                result, fill, state, reason = cleanup(patch, local_bbox, args.strength, previous if args.ema else None)
                previous = state
                if state is not None:
                    # Only transport the sparse change, never resample the original output canvas.
                    delta = np.zeros((h, w, 3), np.float32)
                    delta[top:bottom, left:right] = result.astype(np.float32) - patch
                    alpha = np.zeros((h, w), np.float32)
                    alpha[top:bottom, left:right] = state
                    hard = np.zeros((h, w), np.uint8)
                    hard[top:bottom, left:right] = fill
                    allowed = cv2.warpAffine(hard, matrix, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
                    transported_alpha = cv2.warpAffine(alpha, matrix, (w, h), flags=cv2.INTER_LINEAR) * allowed
                    allowed &= transported_alpha > 0
                    # Recheck the untouched CURRENT canvas after resampling masks.
                    rr, gg, bb = np.moveaxis(before.astype(np.int16), -1, 0)
                    red_now = (rr - gg > 45) | (rr - bb > 65)
                    white_now = (gg >= 105) & (bb >= 85) & (rr - gg <= 65) & ~red_now
                    allowed &= ~red_now & ~white_now
                    transported_delta = cv2.warpAffine(delta, matrix, (w, h), flags=cv2.INTER_LINEAR)
                    after[allowed] = np.rint(before[allowed].astype(np.float32) + transported_delta[allowed]).clip(0, 255).astype(np.uint8)
        else:
            previous = None
        changed = np.any(after != before, axis=2)
        outside = int(np.abs(after.astype(np.int16) - before.astype(np.int16))[~allowed].max(initial=0))
        if outside != 0:
            raise AssertionError('outside-allowed pixels changed')
        for name, image in (('before', cv2.cvtColor(before, cv2.COLOR_RGB2BGR)),
                            ('unfaded', cv2.cvtColor(after, cv2.COLOR_RGB2BGR)), ('allowed', allowed.astype(np.uint8) * 255)):
            if not cv2.imwrite(str(output / name / f'{i:04d}.png'), image):
                raise OSError('PNG write failed')
        records.append({'frame': args.start + i, 'reason': reason, 'changed': int(changed.sum()),
                        'allowed': int(allowed.sum()), 'outside_max': outside, 'outside0': outside == 0,
                        'anchor_to_current': matrix.tolist() if matrix is not None else None,
                        'mask_approved': False})
    # Offline, mask-safe onset/offset ramp. Never retain a correction on a rejected
    # frame: use lookahead within this bounded window rather than copying old teeth.
    weights = transition_weights(np.array([rec['changed'] > 0 for rec in records], dtype=bool))
    for i, before in enumerate(frames):
        budget((n-i) * per_frame)
        weight = float(weights[i])
        raw = cv2.cvtColor(cv2.imread(str(output / 'unfaded' / f'{i:04d}.png')), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(output / 'allowed' / f'{i:04d}.png'), cv2.IMREAD_GRAYSCALE) > 0
        final = before.copy()
        final[mask] = np.rint(before[mask].astype(np.float32)
                             + weight * (raw[mask].astype(np.float32)-before[mask])).clip(0,255).astype(np.uint8)
        assert np.array_equal(final[~mask], before[~mask])
        records[i]['unfaded_changed'] = records[i]['changed']
        records[i]['transition_weight'] = weight
        if weight == 0 and records[i]['unfaded_changed'] > 0:
            records[i]['reason'] = 'transient-island-suppressed'
        records[i]['changed'] = int(np.any(final != before, axis=2).sum())
        if not cv2.imwrite(str(output / 'after' / f'{i:04d}.png'), cv2.cvtColor(final, cv2.COLOR_RGB2BGR)):
            raise OSError('final PNG write failed')
    def encode(command):
        budget(n * h * w * 4)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-threads', '2', '-filter_threads', '2', *command],
                       check=True, timeout=max(0.1, deadline - time.monotonic()))
    common = ['-an', '-c:v', 'libx264', '-crf', '16', '-threads', '2', '-pix_fmt', 'yuv420p']
    # Pad odd canvas edges for yuv420p, never synthesize frames.
    for name in ('before', 'after'):
        encode(['-framerate', str(fps), '-i', str(output / name / '%04d.png'), '-vf',
                'pad=ceil(iw/2)*2:ceil(ih/2)*2', *common, str(output / f'{name}.mp4')])
    encode(['-framerate', str(fps), '-i', str(output / 'before' / '%04d.png'),
            '-framerate', str(fps), '-i', str(output / 'after' / '%04d.png'),
            '-filter_complex_threads', '2', '-filter_complex',
            '[0:v][1:v]hstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2', *common, str(output / 'compare.mp4')])
    for name in ('before', 'after', 'compare'):
        check = cv2.VideoCapture(str(output / f'{name}.mp4'))
        decoded = 0
        try:
            while check.read()[0]:
                budget()
                decoded += 1
                if decoded > n:
                    raise AssertionError('unexpected encoded frames')
        finally:
            check.release()
        if decoded != n:
            raise AssertionError('encoded segment incomplete')
    repo = Path(__file__).resolve().parents[1]
    sha = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True,
                         timeout=max(0.1, deadline - time.monotonic())).stdout.strip()
    files = ['upper_teeth/__init__.py', 'upper_teeth/gap_cleanup.py', 'scripts/probe_upper_gap.py']
    if sha != initial_sha or any(hashlib.sha256((repo / p).read_bytes()).hexdigest() != v
                                 for p, v in initial_hashes.items()):
        raise RuntimeError('code changed during processing')
    metrics = {'status': 'completed_experimental_not_visual_approval', 'source-role': 'final-canvas', 'input_sha256': digest.hexdigest(), 'codeSHA': sha,
               'code_file_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files},
               'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'input_frames': count, 'segment_frames': n, 'fps': fps, 'mask_approved': False,
               'training_enabled': False, 'temporal_validated': False, 'frames': records,
               'scope': 'experimental photometric candidate; not semantic segmentation or tooth structure',
               'encoding': 'silent; outside0 applies to PNG, not lossy MP4'}
    budget()
    with (output / 'metrics.json').open('x', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
