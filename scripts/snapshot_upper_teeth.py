"""Bounded production-pipeline A0 snapshot, not a tooth optimizer or trainer."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELECTED = (225, 228, 251, 252, 264, 266, 292, 446, 456)
LIMIT = 2**30
RESERVE = 64 * 2**20


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def inside(path, root, exists=True):
    result = Path(path).resolve(strict=exists)
    require(result.is_relative_to(root), 'path escapes workspace: ' + str(result))
    return result


def array_hash(value):
    a = np.ascontiguousarray(value)
    require(a.dtype.kind in 'buif', 'only numeric arrays may be snapshotted')
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def validate_snapshot(arrays):
    shapes = {'source_input': (1, 3, 256, 256), 'F': (1, 32, 16, 64, 64),
              'x_s': (1, 21, 3), 'final_k': (581, 1, 21, 3)}
    require(set(arrays) == set(shapes), 'snapshot keys mismatch')
    for key, shape in shapes.items():
        a = arrays[key]
        require(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(),
                'invalid snapshot ' + key)
    return {k: array_hash(v) for k, v in arrays.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=30).strip()


def code_check():
    pin = os.environ.get('PROBE_CODE_SHA', '')
    head = command('git', 'rev-parse', 'HEAD')
    require(re.fullmatch('[0-9a-f]{40}', pin) and pin == head, 'PROBE_CODE_SHA mismatch')
    require(not command('git', 'status', '--porcelain', '--untracked-files=all'), 'dirty Git tree')
    return head


def budget_values(total, used, free, reserve=RESERVE):
    require(0 <= used <= LIMIT - reserve, '1GiB cumulative budget/headroom exceeded')
    headroom = LIMIT - used
    require(total >= used and total + headroom <= 20 * 2**30, '20GiB workspace/headroom exceeded')
    require(free >= headroom, 'insufficient free space')
    return {'workspace_allocated_bytes': total, 'budget_root_allocated_bytes': used,
            'reserved_bytes': reserve, 'remaining_bytes': headroom}


def budget(workspace, budget_root):
    def du(p):
        return int(command('du', '-s', '-B1', str(p)).split()[0]) if p.exists() else 0
    return budget_values(du(workspace), du(budget_root), shutil.disk_usage(workspace).free)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'source', 'driving', 'output', 'budget-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--source-cache', help='Verified same-source snapshot; never reuse driving motion/K')
    p.add_argument('--max-new-mib', type=int, choices=(512, 1024), default=1024)
    p.add_argument('--authorize-experimental', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def check_paths(args):
    workspace = Path(args.workspace).resolve(strict=True)
    require(workspace.is_dir(), 'workspace must be a directory')
    inside(ROOT, workspace)
    source = inside(args.source, workspace)
    driving = inside(args.driving, workspace)
    require(source.is_file() and source.suffix.lower() in ('.png', '.jpg', '.jpeg'), 'source must be an image')
    require(driving.is_file() and driving.suffix.lower() == '.mp4', 'driving must be MP4')
    base = inside(args.budget_root, workspace, exists=False)
    output = inside(args.output, workspace, exists=False)
    require(output != base and output.is_relative_to(base), 'output must be below budget-root')
    require(not base.is_relative_to(ROOT) and not ROOT.is_relative_to(base), 'budget-root must be separate from code')
    require(not source.is_relative_to(base) and not driving.is_relative_to(base), 'inputs must be outside budget-root')
    if args.source_cache:
        cache = inside(args.source_cache, workspace)
        require(cache.is_dir() and not output.is_relative_to(cache) and not cache.is_relative_to(output), 'source-cache/output overlap')
    return workspace, source, driving, base, output


def check_driver(info):
    stream = check_video(info)
    require(0 < min(stream['width'], stream['height']) and max(stream['width'], stream['height']) <= 1280,
            'driver dimensions must be positive and maxdim <=1280')
    return stream


def load_source_cache(directory, workspace, source, code):
    # Numeric/source-only return contract: old final_k is validated but never returned.
    from scripts.probe_upper_tail_capacity import bounded_npz
    directory = inside(directory, workspace)
    manifest = {}
    def file(name):
        path = inside(directory / name, directory)
        manifest[str(path)] = sha256(path)
        return path
    path = file('report.json')
    require(path.stat().st_size <= 32 * 2**20, 'cache report size limit')
    report = json.loads(path.read_text(encoding='utf-8'))
    supervisor = json.loads(file('supervisor.json').read_text(encoding='utf-8'))
    require(report['status'] == supervisor['status'] == 'completed' and report['supervisor_verified'] is True and
            supervisor['returncode'] == 0 and report['code_sha'] == code and
            report['weights_before'] == report['weights_after'] and report['base_before'] == report['base_after'] and
            report['inputs_before'] == report['inputs_after'], 'source cache not verified/immutable/same code')
    require(sha256(source) == report['inputs_before'][report['ArgumentConfig']['source']], 'source cache photo SHA differs')
    for name in ('source_snapshot.npz', 'source_crop.npz', 'source_canvas.png'):
        require(sha256(file(name)) == report['files'][name], 'source cache file hash differs: ' + name)
    values = bounded_npz(directory / 'source_snapshot.npz', 12, {'source_input', 'F', 'x_s', 'final_k'})
    require(validate_snapshot(values) == report['snapshot_arrays'], 'source cache array mismatch')
    crop = bounded_npz(directory / 'source_crop.npz', 16, set(report['crop_arrays']))
    for key, value in crop.items():
        meta = report['crop_arrays'][key]
        require(value.dtype.kind in 'buif' and np.isfinite(value).all() and list(value.shape) == meta['shape'] and
                str(value.dtype) == meta['dtype'] and array_hash(value) == meta['array_sha256'], 'cache crop mismatch')
    return {'arrays': {k: values[k] for k in ('source_input', 'F', 'x_s')}, 'crop': crop,
            'canvas_path': directory / 'source_canvas.png', 'report': report, 'files': manifest}


def supervisor_status(returncode, parent_verified):
    return 'completed' if returncode == 0 and parent_verified else 'failed'


def media(path, audio=False):
    options = ('-select_streams', 'a:0', '-show_packets', '-show_data_hash', 'sha256') if audio else ('-select_streams', 'v:0', '-count_frames')
    return json.loads(command('ffprobe', '-v', 'error', '-threads', '4', *options, '-show_streams', '-of', 'json', str(path)))


def check_video(info, square=False):
    require(len(info['streams']) == 1, 'need one selected video stream')
    s = info['streams'][0]
    require(int(s['nb_read_frames']) == 581 and s['avg_frame_rate'] == '25/1'
            and abs(float(s['duration']) - 23.24) < .0001, 'expected 581 frames / 25fps / 23.24s')
    if square:
        require(s['width'] == s['height'] == 1024, 'expected original 1024 square driver')
    return s


def audio_signature(info):
    require(len(info['streams']) == 1 and info.get('packets'), 'original audio required')
    s = info['streams'][0]
    keys = ('pts', 'dts', 'duration', 'pts_time', 'dts_time', 'duration_time', 'size', 'data_hash')
    packets = [{k: p[k] for k in keys} for p in info['packets']]
    return {'stream': {k: s.get(k) for k in ('codec_name', 'sample_rate', 'channels', 'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration')},
            'packets': packets}


def worker(args, workspace, source, driving, base, output):
    started = time.monotonic()
    import torch
    import cv2
    import imageio
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src import live_portrait_pipeline as pipeline_module
    from src.utils.crop import prepare_paste_back, paste_back

    torch.manual_seed(20260911)
    np.random.seed(20260911)
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
    inputs_before = {str(p): sha256(p) for p in (source, driving)}
    driver_video = check_driver(media(driving))
    cache = load_source_cache(args.source_cache, workspace, source, code) if args.source_cache else None
    cached_canvas = np.asarray(Image.open(cache['canvas_path']).convert('RGB')) if cache else None
    if cache:
        require(array_hash(cached_canvas) == cache['report']['source_canvas']['array_sha256'], 'cached canvas differs')
    audio_before = audio_signature(media(driving, audio=True))
    ac = ArgumentConfig(source=str(source), driving=str(driving), output_dir=str(output))
    explicit = dict(flag_relative_motion=True, flag_stitching=True, flag_normalize_lip=True,
                    driving_multiplier=1.0, flag_eye_retargeting=False, flag_lip_retargeting=False,
                    flag_source_video_eye_retargeting=False, flag_crop_driving_video=False,
                    driving_option='expression-friendly', animation_region='all',
                    flag_use_half_precision=True, flag_do_torch_compile=False,
                    flag_force_cpu=False, flag_do_crop=True, flag_pasteback=False)
    for k, v in explicit.items():
        setattr(ac, k, v)
    cfg, cropcfg = InferenceConfig(), CropConfig()
    for k, v in vars(ac).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if hasattr(cropcfg, k):
            setattr(cropcfg, k, v)
    weights = [inside(getattr(cfg, 'checkpoint_' + k), workspace) for k in 'FMWGS']
    weights_before = {str(p): sha256(p) for p in weights}
    inside(cfg.models_config, workspace)
    inside(cropcfg.landmark_ckpt_path, workspace)
    buffalo = inside(Path(cropcfg.insightface_root) / 'models/buffalo_l', workspace)
    # This deployment intentionally has only the detector and 106-point landmark.
    # Recognition, gender/age and 3D landmark models are not used by this pipeline.
    for name in ('2d106det.onnx', 'det_10g.onnx'):
        require(inside(buffalo / name, workspace).is_file(),
                'existing detector and landmark required; no downloads permitted')
    for model_file in buffalo.glob('*.onnx'):
        inside(model_file, workspace)

    def arr(x):
        return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.array(x, copy=True)

    def serialize(x):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            a = arr(x)
            return {'shape': list(a.shape), 'dtype': str(a.dtype), 'array_sha256': array_hash(a)}
        if dataclasses.is_dataclass(x):
            return {f.name: serialize(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {str(k): serialize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [serialize(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        require(x is None or isinstance(x, (str, int, float, bool)), 'unsupported metadata type')
        return x

    arrays, records, crop_arrays = {}, [], {}
    hooks, encoder = [], None
    frame_index = 0
    canvas = mask = None
    selected_hashes, motion_inputs, template_meta = {}, [], {}
    load_checks = []

    def patch(obj, name, replacement):
        # Restore instance lookup exactly (rather than leaving bound methods in __dict__).
        owned = name in vars(obj)
        old = getattr(obj, name)
        hooks.append((obj, name, old, owned))
        setattr(obj, name, replacement)
        return old

    original_load = torch.nn.Module.load_state_dict

    def checked_load(model, state, *pos, **kw):
        result = original_load(model, state, *pos, **kw)
        require(not result.missing_keys and not result.unexpected_keys, 'non-exact state_dict load')
        load_checks.append(type(model).__name__)
        return result

    def state_hash(wrapper):
        modules = {k: getattr(wrapper, k) for k in ('appearance_feature_extractor', 'motion_extractor', 'warping_module', 'spade_generator')}
        modules.update({'retarget:' + k: v for k, v in wrapper.stitching_retargeting_module.items()})
        return {k: {'parameters': {n: array_hash(arr(t)) for n, t in m.named_parameters()},
                    'buffers': {n: array_hash(arr(t)) for n, t in m.named_buffers()}}
                for k, m in modules.items()}

    try:
        patch(torch.nn.Module, 'load_state_dict', checked_load)
        pipe = pipeline_module.LivePortraitPipeline(cfg, cropcfg)
        w = pipe.live_portrait_wrapper
        require(len(load_checks) >= 7, 'missing base/retarget state load evidence')
        state_before = state_hash(w)
        if cache:
            require(state_before == cache['report']['base_before'] and weights_before == cache['report']['weights_before'], 'cache base/weights differ')
        driving_crop_calls = []
        original_driving_crop = pipe.cropper.crop_driving_video
        def driving_crop_hook(*a, **kw):
            driving_crop_calls.append(True)
            return original_driving_crop(*a, **kw)
        patch(pipe.cropper, 'crop_driving_video', driving_crop_hook)
        original_crop = pipe.cropper.crop_source_image

        def crop_hook(image, *a, **kw):
            nonlocal canvas, mask
            require(canvas is None, 'source may only be detected once')
            if cache:
                require(np.array_equal(image, cached_canvas), 'source canvas does not match cache')
                result = {k: v.copy() for k, v in cache['crop'].items()}
            else:
                result = original_crop(image, *a, **kw)
            require(result is not None, 'source face not detected')
            canvas = image.copy()
            for k, v in result.items():
                if v is not None:
                    numeric = arr(v)
                    array_hash(numeric)
                    crop_arrays[k] = numeric
            require(all(k in crop_arrays for k in ('M_o2c', 'M_c2o', 'img_crop', 'img_crop_256x256', 'lmk_crop')), 'incomplete source crop')
            mask = prepare_paste_back(cfg.mask_crop, result['M_c2o'], dsize=(canvas.shape[1], canvas.shape[0]))
            Image.fromarray(canvas).save(output / 'source_canvas.png')
            return result

        patch(pipe.cropper, 'crop_source_image', crop_hook)
        original_extract = w.extract_feature_3d

        def extract_hook(x):
            require('F' not in arrays, 'source F must be computed once')
            if cache:
                require(array_hash(arr(x)) == array_hash(cache['arrays']['source_input']), 'prepared source input differs from cache')
                result = torch.from_numpy(cache['arrays']['F']).to(device=x.device)
            else:
                result = original_extract(x)
            arrays.update(source_input=arr(x), F=arr(result))
            return result

        patch(w, 'extract_feature_3d', extract_hook)
        original_template = pipe.make_motion_template

        def template_hook(images, *a, **kw):
            require(len(images) == 581 and not motion_inputs, 'expected one complete driving template')
            motion_inputs.extend(array_hash(arr(x)) for x in images)
            template = original_template(images, *a, **kw)
            flat = {f'motion_{i:04d}_{k}': arr(v) for i, item in enumerate(template['motion']) for k, v in item.items()}
            flat.update(c_eyes_lst=np.asarray(template['c_eyes_lst']), c_lip_lst=np.asarray(template['c_lip_lst']))
            np.savez(output / 'driving_motion.npz', **flat)
            template_meta.update(serialize(template))
            return template

        patch(pipe, 'make_motion_template', template_hook)
        original_features = w.spade_generator.forward_features

        def features_hook(*a, **kw):
            h = original_features(*a, **kw)
            if frame_index in SELECTED:
                value = arr(h)
                require(value.shape == (1, 64, 256, 256) and value.dtype == np.float16, 'expected production FP16 H')
                np.savez(output / f'H_f{frame_index:04d}.npz', H=value)
                selected_hashes[str(frame_index)] = serialize(value)
            return h

        patch(w.spade_generator, 'forward_features', features_hook)
        original_warp = w.warp_decode
        final_keys = []

        def warp_hook(f, xs, final_k):
            nonlocal frame_index, encoder
            require(frame_index < 581 and canvas is not None, 'invalid frame sequence')
            if frame_index % 25 == 0:
                budget(workspace, base)
            require(array_hash(arr(f)) == array_hash(arrays['F']), 'source feature changed')
            if cache:
                require(array_hash(arr(xs)) == array_hash(cache['arrays']['x_s']), 'source xs differs from cache; no tolerance')
            if 'x_s' not in arrays:
                arrays['x_s'] = arr(xs)
            require(np.array_equal(arr(xs), arrays['x_s']), 'source keypoints changed')
            final_keys.append(arr(final_k))
            # Do not split W/G contexts: original wrapper owns no_grad and production autocast.
            result = original_warp(f, xs, final_k)
            raw = w.parse_output(result['out'])[0]
            require(raw.shape == (512, 512, 3) and raw.dtype == np.uint8, 'unexpected original parse_output')
            full = paste_back(raw, crop_arrays['M_c2o'], canvas, mask)
            require(full.shape == canvas.shape and full.dtype == np.uint8, 'invalid pasteback')
            if encoder is None:
                height, width = full.shape[:2]
                encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4',
                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                    '-framerate', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
                    '-crf', '18', '-pix_fmt', 'yuv420p', str(output / 'A0_full_silent.mp4')], stdin=subprocess.PIPE)
            encoder.stdin.write(np.ascontiguousarray(full).tobytes())
            records.append({'frame': frame_index, 'final_k': array_hash(final_keys[-1]),
                            'raw512': array_hash(raw), 'full_rgb': array_hash(full)})
            if frame_index in SELECTED:
                Image.fromarray(raw).save(output / f'raw512_f{frame_index:04d}.png')
            if frame_index in (0, 266, 446):
                Image.fromarray(full).save(output / f'full_f{frame_index:04d}.png')
            frame_index += 1
            return result

        patch(w, 'warp_decode', warp_hook)
        dumped = []

        def intercept_dump(path, template):
            require(Path(path) == driving.with_suffix('.pkl') and not dumped, 'unexpected pipeline dump')
            require(serialize(template) == template_meta, 'dump template differs')
            dumped.append({'suppressed_path': str(path), 'template_recorded': True})

        patch(pipeline_module, 'dump', intercept_dump)
        patch(pipeline_module, 'images2video', lambda *a, **kw: None)
        patch(pipeline_module, 'concat_frames', lambda *a, **kw: None)
        patch(pipeline_module, 'has_audio_stream', lambda *a, **kw: False)
        pipe.execute(ac)
        require(frame_index == 581 and len(dumped) == 1 and len(selected_hashes) == 9, 'incomplete pipeline output')
        actual_driving_auto_crop = bool(driving_crop_calls)
        require(len(driving_crop_calls) == int(driver_video['width'] != driver_video['height']), 'unexpected original driving auto-crop path')
        encoder.stdin.close()
        require(encoder.wait(timeout=60) == 0, 'stream encoder failed')
        arrays['final_k'] = np.stack(final_keys)
        snapshot_hashes = validate_snapshot(arrays)
        np.savez(output / 'source_snapshot.npz', **arrays)
        np.savez(output / 'source_crop.npz', **crop_arrays)
        state_after = state_hash(w)
        require(state_before == state_after, 'base parameters/buffers changed')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(output / 'A0_full_silent.mp4'),
                        '-i', str(driving), '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy',
                        '-copyts', '-avoid_negative_ts', 'disabled', str(output / 'A0_full.mp4')], check=True, timeout=60)
        verification = {}
        for name in ('A0_full_silent.mp4', 'A0_full.mp4'):
            verification[name] = check_video(media(output / name))
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-threads', '4', '-i', str(output / name),
                            '-f', 'null', '-'], check=True, timeout=90)
        audio_after = audio_signature(media(output / 'A0_full.mp4', audio=True))
        require(audio_before == audio_after, 'audio packet hash/PTS/DTS/duration mismatch')
        weights_after = {str(p): sha256(p) for p in weights}
        inputs_after = {str(p): sha256(p) for p in (source, driving)}
        require(weights_before == weights_after and inputs_before == inputs_after, 'input/weight files changed')
        require(code_check() == code, 'code changed')
        if cache:
            require(all(sha256(p) == h for p, h in cache['files'].items()), 'source cache changed during run')
        report = {'status': 'completed', 'kind': 'A0 production snapshot; not tooth optimization', 'code_sha': code,
                  'cfg': serialize(cfg), 'cropcfg': serialize(cropcfg), 'ArgumentConfig': serialize(ac),
                  'weights_before': weights_before, 'weights_after': weights_after,
                  'inputs_before': inputs_before, 'inputs_after': inputs_after,
                  'base_before': state_before, 'base_after': state_after, 'state_load_checks': load_checks,
                  'snapshot_arrays': snapshot_hashes, 'crop_arrays': serialize(crop_arrays),
                  'source_canvas': serialize(canvas), 'selected_H': selected_hashes,
                  'motion_input_hashes': motion_inputs, 'template': template_meta, 'suppressed_dump': dumped,
                  'frames': records, 'nframes': frame_index, 'driver_video': driver_video,
                  'actual_driving_auto_crop': actual_driving_auto_crop,
                  'source_cache': None if cache is None else {'directory': str(Path(args.source_cache).resolve()),
                      'files_before': cache['files'], 'files_after': {p: sha256(p) for p in cache['files']},
                      'source_only': True, 'source_redetected': False, 'source_F_recomputed': False,
                      'source_arrays_exact': {k: array_hash(arrays[k]) == array_hash(v) for k, v in cache['arrays'].items()}},
                  'max_new_mib': args.max_new_mib,
                  'video_verification': verification, 'full_decode_passed': True,
                  'audio_before': audio_before, 'audio_after': audio_after, 'audio_exact': True,
                  'files': {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name not in ('report.json', 'supervisor.json')},
                  'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                  'numpy': np.__version__, 'cv2': cv2.__version__, 'imageio': imageio.__version__,
                                  'ffmpeg': command('ffmpeg', '-version').splitlines()[0],
                                  'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
                                  'cpu_affinity': sorted(os.sched_getaffinity(0)),
                                  'seed': 20260911, 'cudnn_benchmark': False,
                                  'deterministic_algorithms': True, 'tf32': False},
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'wall_seconds': time.monotonic() - started, **budget(workspace, base)}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    finally:
        for obj, name, old, owned in reversed(hooks):
            if owned:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


class OwnedProcessGroup:
    """Keep the leader unreaped until group cleanup, so its PGID cannot be reused."""
    def __init__(self, process):
        self.process = process
        self.retained = True
        self.lock = threading.Lock()

    def exited(self):
        return os.waitid(os.P_PID, self.process.pid,
                         os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def _kill(self):
        if self.retained:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def kill(self):
        with self.lock:
            self._kill()

    def finish(self):
        with self.lock:
            if self.retained:
                self._kill()
                self.process.wait(timeout=30)
                self.retained = False


def main():
    global LIMIT
    started = time.monotonic()
    args = parser().parse_args()
    LIMIT = args.max_new_mib * 2**20
    require(sys.platform == 'linux' and args.authorize_experimental, 'Linux and explicit authorization required')
    workspace, source, driving, base, output = check_paths(args)
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)), 'missing isolated ' + key)
        require(inside(os.environ[key], workspace).is_dir(), 'cache must exist')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    require(os.environ.get('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT') == '1', 'ImageIO must inherit the owned process group')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'select one explicit CUDA UUID')
    code_check()
    if args._worker:
        require(os.environ.get('UPPER_SNAPSHOT_PARENT') == str(os.getppid()), 'worker must be supervised')
        # Only this owned worker and its descendants; never change another process.
        allowed_cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed_cpus[:4])
        worker(args, workspace, source, driving, base, output)
        return
    require(not output.exists() and not Path(args.output).is_symlink(),
            'refuse existing output; never delete artifacts')
    import fcntl
    sys.path.insert(0, str(ROOT))
    from scripts.run_contact_release import elapsed_charge, read_json, write_json
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'source-mouth.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = elapsed_charge(base)
        require(previous + 600 <= 1800, 'cumulative 1800s allocation exhausted')
        initial_budget = budget(workspace, base)
        output.mkdir(parents=True)
        write_json(output / 'supervisor.json', {'status': 'running', 'stage': 'snapshot'})
        process = owned = timer = None
        verified, error = False, None
        try:
            env = dict(os.environ, UPPER_SNAPSHOT_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[key] = '4'
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                       cwd=ROOT, env=env, start_new_session=True)
            owned = OwnedProcessGroup(process)
            timer = threading.Timer(max(0, 585 - (time.monotonic() - started)), owned.kill)
            timer.daemon = True
            timer.start()
            while not owned.exited():
                require(time.monotonic() - started < 585, 'worker timeout; final audit margin reserved')
                budget(workspace, base)
                time.sleep(.5)
            owned.finish()
            require(process.returncode == 0, 'snapshot worker failed; partial output is not accepted')
            report = read_json(output / 'report.json', 32 * 2**20)
            require(report['status'] == 'completed' and report['full_decode_passed'] is True, 'worker snapshot incomplete')
            report.update(budget_before=initial_budget, budget_after=budget(workspace, base))
            write_json(output / 'report.json', report)
            verified = True
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if owned is not None:
                owned.finish()
            wall = time.monotonic() - started
            verified = verified and wall <= 600 and previous + wall <= 1800
            rc = None if process is None else process.returncode
            status = supervisor_status(rc, verified)
            record = {'status': status, 'stage': 'snapshot', 'returncode': rc, 'error': error,
                      'wall_seconds': wall, 'cumulative_wall_seconds': previous + wall, 'hard_limit_seconds': 600}
            path = output / 'report.json'
            try:
                report = read_json(path, 32 * 2**20) if path.exists() else {}
            except (ValueError, OSError):
                report = {}
            report.update(status=status, supervisor_verified=verified, wall_seconds=wall, supervisor=record)
            write_json(path, report)
            write_json(output / 'supervisor.json', record)
        require(verified, 'snapshot final supervisor audit failed')


if __name__ == '__main__':
    main()
