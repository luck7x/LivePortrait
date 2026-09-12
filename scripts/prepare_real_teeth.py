"""Bounded real-video aligned reconstruction pairs; no training or tooth edits.

CLI roles are declarations: clip118=train, clip80=validation. Different file hashes
are enforced, not identity separation. Landmark semantics require visual review.
"""
import argparse
import dataclasses
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

SELECTED = tuple(sorted(set(range(200, 301)) | set(range(442, 472)) | {0, 80, 116, 348, 580}))
OVERLAYS = (0, 225, 228, 251, 255, 266, 446)
ROI = (160, 290, 350, 410)  # x0,y0,x1,y1 in native 512 feature coordinates
SEED = 20260913


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'train-video', 'validation-video', 'output', 'budget-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--authorize-real', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def check_paths(args):
    workspace = Path(args.workspace).resolve(strict=True)
    require(workspace.is_dir(), 'workspace must be a directory')
    inside(ROOT, workspace)
    videos = [inside(p, workspace) for p in (args.train_video, args.validation_video)]
    require(all(p.is_file() and p.suffix.lower() == '.mp4' for p in videos), 'inputs must be MP4 files')
    base = inside(args.budget_root, workspace, exists=False)
    output = inside(args.output, workspace, exists=False)
    require(output != base and output.is_relative_to(base), 'output must be below budget-root')
    require(not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'budget-root must be separate from code')
    require(all(not p.is_relative_to(base) for p in videos), 'inputs must be outside budget-root')
    require(videos[0] != videos[1], 'videos must differ')
    return workspace, videos, base, output


def check_input_media(info):
    s = check_video(info)
    require(0 < min(s['width'], s['height']) and max(s['width'], s['height']) <= 1280,
            'source dimensions must be positive and at most 1280')
    return s


def validate_feature(a):
    require(a.shape == (1, 16, 120, 190) and a.dtype == np.float16 and np.isfinite(a).all(),
            'invalid native feature ROI')
    require(a.nbytes < 2**20, 'feature array exceeds 1MiB')
    return array_hash(a)


def validate_landmarks(a, count=None):
    shape = (203, 2) if count is None else (count, 203, 2)
    require(a.shape == shape and a.dtype.kind == 'f' and np.isfinite(a).all(), 'invalid 203 landmarks')
    return array_hash(a)


def alignment_review(gt, baseline, nose_indices):
    """Eye indices follow crop.py; spatial nose proxy is explicitly unapproved."""
    groups = {'left_eye': [0, 6, 12, 18], 'right_eye': [24, 30, 36, 42],
              'nose_spatial_proxy_unverified': list(nose_indices)}
    result = {}
    for name, indices in groups.items():
        if not indices:
            result[name] = {'indices': [], 'available': False}
            continue
        g, b = gt[indices].mean(axis=0), baseline[indices].mean(axis=0)
        result[name] = {'indices': indices, 'gt_center': g.tolist(), 'base_center': b.tolist(),
                        'center_distance_px': float(np.linalg.norm(g - b))}
    return result


def file_inventory(output):
    return {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(output.rglob('*')) if p.is_file() and p != output / 'report.json'}


def encoder(path):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '512x512', '-framerate', '25',
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '18',
        '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)


def verify_outputs(directory, video, original_audio):
    verification = {}
    for kind in ('GT', 'BASE'):
        silent = directory / f'{kind}_aligned_full_silent.mp4'
        full = directory / f'{kind}_aligned_full.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(silent), '-i', str(video),
            '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-copyts', '-avoid_negative_ts', 'disabled',
            str(full)], check=True, timeout=60)
        for path in (silent, full):
            stream = check_video(media(path))
            require(stream['width'] == stream['height'] == 512, 'aligned output must be 512 square')
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-threads', '4', '-i', str(path),
                            '-f', 'null', '-'], check=True, timeout=90)
            verification[path.name] = {'stream': stream, 'full_decode_passed': True}
        after = audio_signature(media(full, audio=True))
        require(after == original_audio, 'all audio packet hashes/timestamps must match')
        verification[full.name].update(audio_exact=True, audio_signature=after)
    return verification


def worker(workspace, videos, base, output):
    started = time.monotonic()
    import torch
    import torch.nn.functional as functional
    import cv2
    import imageio
    from PIL import Image, ImageDraw
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.utils.crop import _transform_img

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    cv2.setNumThreads(4)
    require(torch.cuda.device_count() == 1, 'exactly one visible CUDA device required')
    code = code_check()
    before_inputs = {str(p): sha256(p) for p in videos}
    require(len(set(before_inputs.values())) == 2, 'train and validation hashes must differ')
    video_infos = [check_input_media(media(p)) for p in videos]
    audios = [audio_signature(media(p, audio=True)) for p in videos]
    cfg, cropcfg = InferenceConfig(), CropConfig()
    for name in ('flag_relative_motion', 'flag_stitching', 'flag_normalize_lip', 'flag_eye_retargeting',
                 'flag_lip_retargeting', 'flag_source_video_eye_retargeting', 'flag_pasteback',
                 'flag_do_torch_compile', 'flag_force_cpu'):
        setattr(cfg, name, False)
    cfg.flag_use_half_precision = True
    cfg.device_id = cropcfg.device_id = 0
    cropcfg.dsize = 512
    models = [inside(getattr(cfg, 'checkpoint_' + k), workspace) for k in 'FMWGS']
    models += [inside(cropcfg.landmark_ckpt_path, workspace), inside(cfg.models_config, workspace)]
    buffalo = inside(Path(cropcfg.insightface_root) / 'models/buffalo_l', workspace)
    for name in ('det_10g.onnx', '2d106det.onnx'):
        require(inside(buffalo / name, workspace).is_file(), 'existing ONNX required; no download')
    models += [inside(p, workspace) for p in sorted(buffalo.glob('*.onnx'))]
    require(all(p.is_file() for p in models), 'existing model files required')
    before_models = {str(p): sha256(p) for p in models}

    def arr(t):
        return t.detach().cpu().numpy().copy() if torch.is_tensor(t) else np.array(t, copy=True)

    def serialize(value):
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            a = arr(value)
            return {'shape': list(a.shape), 'dtype': str(a.dtype), 'array_sha256': array_hash(a)}
        if dataclasses.is_dataclass(value):
            return {f.name: serialize(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {str(k): serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialize(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        require(value is None or isinstance(value, (str, int, float, bool)), 'unsupported metadata')
        return value

    loads = []
    original_load = torch.nn.Module.load_state_dict

    def strict_load(module, state, *args, **kwargs):
        require((not args or args[0] is True) and kwargs.get('strict', True) is True, 'strict loading required')
        result = original_load(module, state, *args, **kwargs)
        require(not result.missing_keys and not result.unexpected_keys, 'checkpoint incompatibility')
        loads.append(type(module).__name__)
        return result

    torch.nn.Module.load_state_dict = strict_load
    try:
        w = LivePortraitWrapper(cfg)
        cropper = Cropper(crop_cfg=cropcfg, device_id=0, flag_force_cpu=False)
    finally:
        torch.nn.Module.load_state_dict = original_load
    require(len(loads) >= 7, 'missing strict FMWGS load evidence')

    def state_hash():
        modules = {k: getattr(w, k) for k in ('appearance_feature_extractor', 'motion_extractor',
                   'warping_module', 'spade_generator')}
        modules.update({'retarget:' + k: v for k, v in w.stitching_retargeting_module.items()})
        require(all(not m.training for m in modules.values()), 'base models must remain eval')
        return {k: {'parameters': {n: array_hash(arr(t)) for n, t in m.named_parameters()},
                    'buffers': {n: array_hash(arr(t)) for n, t in m.named_buffers()}}
                for k, m in modules.items()}

    before_state = state_hash()
    reports = {}
    for split, role, video, info, audio in zip(('train', 'validation'), ('clip118', 'clip80'), videos, video_infos, audios):
        directory = output / split
        directory.mkdir()
        reader = imageio.get_reader(str(video), 'ffmpeg')  # identical sequential RGB decoder to io.load_video
        writers = []
        g = w.spade_generator
        require('forward_features' not in vars(g), 'unexpected pre-existing generator override')
        original_features = g.forward_features
        frame_index = -1
        feature_roi = None
        feature_calls = 0

        def capture_features(*args, **kwargs):
            nonlocal feature_roi, feature_calls
            h = original_features(*args, **kwargs)
            if frame_index in SELECTED:
                require(feature_roi is None, 'duplicate feature call in one frame')
                require(tuple(h.shape) == (1, 64, 256, 256) and h.dtype == torch.float16, 'expected production H')
                # Inspect the original call, do not rerun W/G or retain the full H.
                shuffled = functional.pixel_shuffle(functional.leaky_relu(h, .2), 2)
                x0, y0, x1, y1 = ROI
                feature_roi = arr(shuffled[:, :, y0:y1, x0:x1])
                validate_feature(feature_roi)
                feature_calls += 1
            return h

        g.forward_features = capture_features
        try:
            records, keys, gt_lmks, base_lmks, eyes, lips, selected_records = [], [], [], [], [], [], []
            writers = [encoder(directory / f'{kind}_aligned_full_silent.mp4') for kind in ('GT', 'BASE')]
            for i, rgb in enumerate(reader):
                require(i < 581 and rgb.shape == (info['height'], info['width'], 3) and rgb.dtype == np.uint8,
                        'invalid sequential source RGB')
                if i % 25 == 0:
                    budget(workspace, base)
                if i == 0:
                    canvas = rgb.copy()
                    crop = cropper.crop_source_image(canvas, cropcfg)  # exactly once per video
                    require(crop is not None, 'source face not detected')
                    crop_m = arr(crop['M_o2c'])
                    inverse_m = arr(crop['M_c2o'])
                    hint = arr(crop['pt_crop'])  # 106 points already in crop coordinates; NOT lmk_crop
                    require(hint.shape == (106, 2) and np.isfinite(hint).all(), 'invalid pt_crop106')
                    Image.fromarray(canvas).save(directory / 'source_canvas.png')
                    Image.fromarray(crop['img_crop']).save(directory / 'source_crop.png')
                    source_input = w.prepare_source(cv2.resize(crop['img_crop'], (256, 256), interpolation=cv2.INTER_AREA))
                    source_info = w.get_kp_info(source_input)
                    xs = w.transform_keypoint(source_info)
                    fs = w.extract_feature_3d(source_input)  # source F computed only once
                    fixed = {'source_input': arr(source_input), 'F': arr(fs), 'x_s': arr(xs)}
                    fixed_hashes = {k: array_hash(v) for k, v in fixed.items()}
                    info_hashes = serialize(source_info)
                    self_raw = w.parse_output(w.warp_decode(fs, xs, xs)['out'])[0]
                gt = _transform_img(rgb, crop_m, 512)  # original INTER_LINEAR, default constant-zero border
                require(gt.shape == (512, 512, 3) and gt.dtype == np.uint8, 'invalid real aligned GT')
                target_input = w.prepare_source(cv2.resize(gt, (256, 256), interpolation=cv2.INTER_AREA))
                target_info = w.get_kp_info(target_input)
                kt = w.transform_keypoint(target_info)  # absolute, no relative/stitch/normalization/retarget
                if i == 0:
                    require(np.array_equal(gt, crop['img_crop']), 'f0 GT differs from source crop')
                    require(np.array_equal(arr(target_input), fixed['source_input']), 'f0 input differs')
                    require(np.array_equal(arr(kt), fixed['x_s']), 'absolute K0 differs from xs')
                frame_index, feature_roi = i, None
                raw = w.parse_output(w.warp_decode(fs, xs, kt)['out'])[0]
                require(raw.shape == (512, 512, 3) and raw.dtype == np.uint8, 'invalid base output')
                if i == 0:
                    require(np.array_equal(raw, self_raw), 'f0 differs from source self reconstruction')
                require(array_hash(arr(fs)) == fixed_hashes['F'] and array_hash(arr(xs)) == fixed_hashes['x_s'],
                        'fixed source F/xs changed')
                keys.append(arr(kt))
                lgt = arr(cropper.human_landmark_runner.run(gt, lmk=hint.copy()))
                validate_landmarks(lgt)
                er, lr = w.calc_ratio([lgt])
                eyes.append(np.asarray(er)[0])
                lips.append(np.asarray(lr)[0])
                if i == 0:
                    eye_center = lgt[[0, 6, 12, 18, 24, 30, 36, 42]].mean(axis=0)
                    mouth_center = lgt[[48, 66]].mean(axis=0)
                    eye_span = np.linalg.norm(lgt[[24, 30, 36, 42]].mean(axis=0) - lgt[[0, 6, 12, 18]].mean(axis=0))
                    # Spatial diagnostic proxy only: no invented authoritative 203 nose mapping.
                    nose_indices = np.flatnonzero((lgt[:, 1] > eye_center[1] + .25 * (mouth_center[1] - eye_center[1])) &
                        (lgt[:, 1] < eye_center[1] + .75 * (mouth_center[1] - eye_center[1])) &
                        (np.abs(lgt[:, 0] - eye_center[0]) < .25 * eye_span)).tolist()
                for proc, image in zip(writers, (gt, raw)):
                    proc.stdin.write(np.ascontiguousarray(image).tobytes())
                record = {'frame': i, 'source_rgb': array_hash(rgb), 'key': array_hash(keys[-1]),
                          'rawGT': array_hash(gt), 'base': array_hash(raw), 'target_input': array_hash(arr(target_input))}
                records.append(record)
                if i in SELECTED:
                    lbase = arr(cropper.human_landmark_runner.run(raw, lmk=hint.copy()))
                    validate_landmarks(lbase)
                    gt_lmks.append(lgt)
                    base_lmks.append(lbase)
                    paths = {}
                    for kind, image in (('GT', gt), ('BASE', raw)):
                        path = directory / f'{kind}_f{i:04d}.png'
                        Image.fromarray(image).save(path)
                        paths[kind] = {'file': path.name, 'sha256': sha256(path), 'array_sha256': array_hash(image)}
                    feature_hash = validate_feature(feature_roi)
                    feature_path = directory / f'feature_f{i:04d}.npz'
                    np.savez(feature_path, feature=feature_roi)
                    require(feature_path.stat().st_size <= 2**20, 'feature file exceeds 1MiB')
                    selected_records.append({'frame': i, 'images': paths, 'feature': feature_path.name,
                        'feature_sha256': sha256(feature_path), 'feature_array_sha256': feature_hash,
                        'alignment_review': alignment_review(lgt, lbase, nose_indices)})
                    if i in OVERLAYS:
                        panels = []
                        for label, image, lm in (('GT real aligned', gt, lgt), ('BASE original model', raw, lbase)):
                            panel = Image.fromarray(image).resize((1024, 1024), Image.Resampling.NEAREST)
                            draw = ImageDraw.Draw(panel)
                            draw.text((10, 10), f'{label} f{i} - indices UNAPPROVED', fill='yellow')
                            for index in range(48, 109):
                                x, y = (lm[index] * 2).tolist()
                                color = 'red' if index in (90, 102, 48, 66) else ('cyan' if 84 <= index < 108 else 'yellow')
                                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
                                draw.text((x+3, y-4), str(index), fill=color)
                            panels.append(panel)
                        overlay = Image.new('RGB', (2048, 1024))
                        overlay.paste(panels[0], (0, 0))
                        overlay.paste(panels[1], (1024, 0))
                        overlay.save(directory / f'landmarks_f{i:04d}.png')
                del target_input, target_info, kt
            require(len(records) == 581 and feature_calls == len(SELECTED), 'incomplete video/features')
            for proc in writers:
                proc.stdin.close()
                require(proc.wait(timeout=60) == 0, 'encoder failed')
            for name, tensor in (('source_input', source_input), ('F', fs), ('x_s', xs)):
                require(array_hash(arr(tensor)) == fixed_hashes[name], 'source tensor changed')
            require(serialize(source_info) == info_hashes, 'source motion info changed')
            fixed['final_k'] = np.stack(keys)
            snapshot_hashes = validate_snapshot(fixed)
            np.savez(directory / 'source.npz', **fixed, cropM=crop_m, M_o2c=crop_m,
                     M_c2o=inverse_m, pt_crop106=hint)
            landmarks = {'lmks_gt': np.stack(gt_lmks), 'lmks_base': np.stack(base_lmks)}
            for value in landmarks.values():
                validate_landmarks(value, len(SELECTED))
            np.savez(directory / 'annotations.npz', selected=np.asarray(SELECTED), **landmarks)
            ratios = {'eyes': np.stack(eyes), 'lip': np.stack(lips)}
            require(all(a.shape[0] == 581 and np.isfinite(a).all() for a in ratios.values()), 'invalid ratios')
            np.savez(directory / 'ratios.npz', **ratios)
            reports[split] = {'declared_role': role, 'input': str(video), 'input_media': info,
                'audio_before': audio, 'video_verification': verify_outputs(directory, video, audio),
                'source_detection_count': 1, 'source_feature_count': 1, 'source_info': info_hashes,
                'source_arrays': snapshot_hashes, 'crop_arrays': serialize({'M_o2c': crop_m, 'M_c2o': inverse_m, 'pt_crop106': hint}),
                'source_canvas': serialize(canvas), 'annotations': serialize(landmarks), 'ratios': serialize(ratios),
                'frames': records, 'selected': selected_records, 'f0_crop_key_self_reconstruction_exact': True,
                'alignment_review_status': 'pending visual review; center differences are not a pass',
                'nose_proxy_indices': nose_indices, 'mouth_indices_human_approved': False}
        finally:
            delattr(g, 'forward_features')
            reader.close()
            for proc in writers:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    after_state = state_hash()
    after_models = {str(p): sha256(p) for p in models}
    after_inputs = {str(p): sha256(p) for p in videos}
    require(before_state == after_state, 'original model parameters/buffers changed')
    require(before_models == after_models and before_inputs == after_inputs, 'models/media changed')
    require(code_check() == code, 'code changed during run')
    report = {'status': 'completed', 'kind': 'real aligned same-video reconstruction supervision; NOT optimization',
        'identity_holdout_claimed': False, 'cross_video_identity_required': False, 'training_performed': False,
        'semantic_labels_approved': False, 'code_sha': code, 'cfg': serialize(cfg), 'cropcfg': serialize(cropcfg),
        'selected_indices': SELECTED, 'feature_roi_xyxy': ROI, 'source_dsize': 512,
        'decoder': 'imageio.get_reader(ffmpeg), sequential original RGB; no recoloring',
        'alignment': 'fixed frame0 M_o2c via original _transform_img: INTER_LINEAR/default zero border',
        'models_before': before_models, 'models_after': after_models, 'inputs_before': before_inputs,
        'inputs_after': after_inputs, 'base_before': before_state, 'base_after': after_state,
        'strict_load_checks': loads, 'videos': reports,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cv2': cv2.__version__, 'imageio': imageio.__version__,
            'ffmpeg': command('ffmpeg', '-version').splitlines()[0], 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
            'gpu_name': torch.cuda.get_device_name(0), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'seed': SEED, 'fp16': True, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'deterministic_algorithms': True, 'tf32': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'IMAGEIO_FFMPEG_NO_PREVENT_SIGINT': os.environ['IMAGEIO_FFMPEG_NO_PREVENT_SIGINT']},
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'worker_wall_seconds': time.monotonic() - started, **budget(workspace, base)}
    report['files'] = file_inventory(output)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    started = time.monotonic()
    args = parser().parse_args()
    require(sys.platform == 'linux' and args.authorize_real, 'Linux and --authorize-real required')
    workspace, videos, base, output = check_paths(args)
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)), 'missing isolated ' + key)
        require(inside(os.environ[key], workspace).is_dir(), 'cache must exist')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'select one CUDA UUID')
    code_check()
    if args._worker:
        require(os.environ.get('REAL_TEETH_PARENT') == str(os.getppid()), 'worker must be supervised')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic CUBLAS required')
        require(os.environ.get('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT') == '1', 'decoder must inherit owned process group')
        cpus = sorted(os.sched_getaffinity(0))
        require(len(cpus) >= 4, 'four isolated worker CPUs required')
        os.sched_setaffinity(0, cpus[:4])
        worker(workspace, videos, base, output)
        return
    require(not base.exists() and not Path(args.budget_root).is_symlink(), 'new budget-root required')
    require(not output.exists() and not Path(args.output).is_symlink(), 'refuse existing output')
    initial_budget = budget(workspace, base)
    output.mkdir(parents=True)
    env = dict(os.environ, REAL_TEETH_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               IMAGEIO_FFMPEG_NO_PREVENT_SIGINT='1')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '4'
    owned = timer = process = None
    failure = None
    completed = False
    try:
        with open(output / 'worker.log', 'xb') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                       cwd=ROOT, env=env, start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
        owned = OwnedProcessGroup(process)
        timer = threading.Timer(max(0, 600 - (time.monotonic() - started)), owned.kill)
        timer.daemon = True
        timer.start()
        while not owned.exited():
            require(time.monotonic() - started < 600, '600s hard timeout')
            budget(workspace, base)
            time.sleep(.5)
        owned.finish()
        require(process.returncode == 0, 'real preparation worker failed; partial output not accepted')
        final_budget = budget(workspace, base)
        report_path = output / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(report['status'] == 'completed', 'missing completed worker report')
        require(time.monotonic() - started < 600, '600s hard timeout')
        report.update(supervisor_verified=True, wall_seconds=time.monotonic() - started,
                      budget_before=initial_budget, budget_after=final_budget)
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        completed = True
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc),
                   'failed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'wall_seconds': time.monotonic() - started}
        (output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if owned is not None:
            owned.finish()
        supervisor = {'status': 'completed' if completed else 'failed',
            'returncode': process.returncode if process is not None else None,
            'wall_seconds': time.monotonic() - started, 'hard_limit_seconds': 600, 'failure': failure}
        (output / 'supervisor.json').write_text(json.dumps(supervisor, indent=2) + '\n', encoding='utf-8')
        if completed:
            report['files'] = file_inventory(output)
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
