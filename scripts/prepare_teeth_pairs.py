"""Prepare aligned self-reenactment A/GT candidates; never approves tooth masks.

Run only in the authorized Linux GPU workspace. Original frames are GT; one
source-frame crop transform aligns the driving input to v1's source crop. The
v1 core renders back to the original canvas, and only a bounded selected window
is captured losslessly before video encoding. This is training-data preparation,
not a new optimized model or an automatically admitted training dataset.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

from diagnose_portrait import REPO, WorkspaceBudget, inside, sha256, write_json


def run(args):
    if sys.platform != 'linux':
        raise RuntimeError('GPU preparation must run on the authorized Linux server')
    root = args.workspace.resolve(strict=True)
    inside(REPO, root)
    video, output = inside(args.video, root), inside(args.output_dir, root)
    if output.exists() or not video.is_file() or video.stat().st_size > 64 * 2**20:
        raise ValueError('Use a bounded input and a new output directory')
    for name in ('HOME', 'TMPDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        if not os.environ.get(name):
            raise RuntimeError('Missing isolated environment ' + name)
        inside(os.environ[name], root)
    meta = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(video)], text=True, timeout=45))
    stream = meta['streams'][0]
    n = int(stream['nb_read_frames'])
    fps = float(__import__('fractions').Fraction(stream['avg_frame_rate']))
    x0, y0, x1, y1 = args.mouth_roi
    if not (0 <= args.first_frame <= args.last_frame < n <= 600
            and args.last_frame-args.first_frame+1 <= 96
            and fps == 25 and float(meta['format']['duration']) <= 25
            and max(stream['width'], stream['height']) <= 1280
            and 0 <= x0 < x1 <= stream['width'] and 0 <= y0 < y1 <= stream['height']
            and x1-x0 <= 256 and y1-y0 <= 256):
        raise ValueError('Unsupported video/window/ROI; this pilot is bounded to 25fps short clips')
    budget = WorkspaceBudget(root, 20)
    selected_count = args.last_frame - args.first_frame + 1
    reserve = (n * (512 * 512 * 3 + 65536)
               + selected_count * (2 * stream['width'] * stream['height'] * 3 + 131072)
               + 128 * 2**20)
    budget.require(reserve)
    sys.path.insert(0, str(REPO))
    import cv2
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
        raise RuntimeError('Expose exactly one approved CUDA GPU')
    output.mkdir(parents=True)
    (output/'inputs').mkdir()
    (output/'a_frames').mkdir()
    (output/'gt_frames').mkdir()
    report = {'status': 'running', 'kind': 'unreviewed_same_person_pair_candidates',
              'source_video_sha256': sha256(video), 'probe': meta,
              'frame_ids': list(range(args.first_frame, args.last_frame+1)),
              'mouth_roi': list(args.mouth_roi), 'source_frame': 0,
              'code_sha': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'training_approved': False, 'mask_reviewed': False, 'paired_gt_available': False}
    write_json(output/'preparation.json', report)
    capture = cv2.VideoCapture(str(video))
    process = None
    pipeline = None
    old_paste = None
    old_crop = None
    try:
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError('Cannot decode source frame')
        first_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(first_rgb).save(output/'inputs/source.png')
        options = ArgumentConfig(source=str(output/'inputs/source.png'),
            driving=str(output/'inputs/aligned.avi'), output_dir=str(output/'videos'),
            flag_normalize_lip=True, flag_relative_motion=True,
            flag_crop_driving_video=False, driving_multiplier=1.0)
        cfg = partial_fields(InferenceConfig, dataclasses.asdict(options))
        crop_cfg = CropConfig()
        for p in (cfg.checkpoint_F, cfg.checkpoint_M, cfg.checkpoint_W, cfg.checkpoint_G,
                  cfg.checkpoint_S, crop_cfg.landmark_ckpt_path, crop_cfg.insightface_root):
            inside(p, root)
        report['arguments'] = dataclasses.asdict(options)
        report['weights'] = {str(inside(p, root).relative_to(root)): sha256(p)
                             for p in (cfg.checkpoint_F, cfg.checkpoint_M, cfg.checkpoint_W,
                                       cfg.checkpoint_G, cfg.checkpoint_S)}
        pipeline = LivePortraitPipeline(cfg, crop_cfg)
        crop = pipeline.cropper.crop_source_image(first_rgb, crop_cfg)
        if crop is None:
            raise RuntimeError('No source face')
        matrix = crop['M_o2c'][:2]
        report['fixed_original_to_crop'] = matrix.tolist()
        old_crop = pipeline.cropper.crop_source_image
        def cached_source_crop(image, config):
            if not np.array_equal(image, first_rgb):
                raise RuntimeError('Source pixels changed before the cached crop')
            return crop
        pipeline.cropper.crop_source_image = cached_source_crop
        report['source_crop_reused_after_pixel_equality_check'] = True
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                   '-pix_fmt', 'rgb24', '-s', '512x512', '-r', '25', '-i', 'pipe:0',
                   '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
                   str(output/'inputs/aligned.avi')]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        expected_crops, gt = {}, []
        for i in range(n):
            if i == 0:
                rgb = first_rgb
            else:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Missing input frame')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aligned = cv2.warpAffine(rgb, matrix, (512, 512), flags=cv2.INTER_LINEAR)
            process.stdin.write(aligned.tobytes())
            if i == 0 or args.first_frame <= i <= args.last_frame:
                expected_crops[i] = hashlib_rgb(aligned)
            if args.first_frame <= i <= args.last_frame:
                budget.require(rgb.nbytes + 65536)
                Image.fromarray(rgb).save(output/'gt_frames'/f'{i:06d}.png')
                gt.append(rgb[y0:y1, x0:x1].copy())
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError('Lossless aligned video encoding failed')
        process = None
        capture.release()
        # Prove that the actual OpenCV decoder used by v1 sees the exact aligned pixels.
        check = cv2.VideoCapture(str(output/'inputs/aligned.avi'))
        try:
            for i in range(n):
                ok, bgr = check.read()
                if not ok:
                    raise RuntimeError('Aligned video frame missing')
                if i in expected_crops and hashlib_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) != expected_crops[i]:
                    raise RuntimeError('Aligned video decoder changed RGB pixels')
        finally:
            check.release()
        budget.require(128 * 2**20)
        old_paste = pipeline_module.paste_back
        bases = []
        count = 0
        def record_paste(*a, **kw):
            nonlocal count
            result = old_paste(*a, **kw)
            if args.first_frame <= count <= args.last_frame:
                budget.require(result.nbytes + 65536)
                Image.fromarray(result).save(output/'a_frames'/f'{count:06d}.png')
                bases.append(result[y0:y1, x0:x1].copy())
            count += 1
            return result
        pipeline_module.paste_back = record_paste
        pipeline.execute(options)
        if count != n or len(bases) != len(gt):
            raise RuntimeError('A/GT frame count mismatch')
        budget.require(sum(a.nbytes for a in bases + gt) + 2**20)
        np.savez_compressed(output/'unreviewed_pairs.npz', base_rgb=np.stack(bases),
                            target_rgb=np.stack(gt), frame_ids=np.array(report['frame_ids'], dtype=np.int64))
        report.update(status='prepared_pending_masks_and_alignment_review',
                      captured_frames=len(bases), aligned_decoder_exact=True,
                      pair_sha256=sha256(output/'unreviewed_pairs.npz'))
        budget.require(0)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        capture.release()
        if old_paste is not None:
            pipeline_module.paste_back = old_paste
        if old_crop is not None:
            pipeline.cropper.crop_source_image = old_crop
        if process is not None:
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        write_json(output/'preparation.json', report)
    return report


def hashlib_rgb(array):
    import hashlib
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--first-frame', type=int, default=442)
    parser.add_argument('--last-frame', type=int, default=471)
    parser.add_argument('--mouth-roi', type=int, nargs=4, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
