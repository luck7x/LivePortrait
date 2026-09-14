"""Fixed source/normalization ablation and historical student replay; no training."""
import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_contact_samples as prep
from scripts.snapshot_upper_teeth import (OwnedProcessGroup, array_hash, audio_signature,
    check_video, code_check, command, inside, media, require, sha256)
from scripts.run_contact_release import (elapsed_charge, read_json, write_json, tracked,
    runtime, load_base, states, ffmpeg, tensor_image, pixel_audit)
from scripts import run_real_teeth as real
from scripts.probe_upper_tail_capacity import bounded_npz

OLD = '55f865447014b0c35bc6aca3bdc32648937b9f8ceb6c4a420a43a9fdf251032c'
NEW = '7aa906e3135b1c7c903c3c1057f4fe6c66bb33f4d59c8390d11258c14329226a'
TWO_DRIVER_SOURCE = 'ded9d6ed7facbf5bf2646e982858d87ffc7ebd45b110a95706df60157f3fbed2'
STUDENT = '32e108b4258cd7c5c6fcf4b4a09cd1e20369a8cddc3183821106d841ac6c2b2c'
SAMPLED = (0, 80, 116, 225, 226, 228, 251, 255, 260, 263, 266, 292, 348, 442, 446, 456, 470, 580)
LIMIT = 512 * 2**20


def space_values(total, used, free):
    require(0 <= used <= LIMIT - 32 * 2**20, '512MiB cumulative outputs/headroom exceeded')
    remaining = LIMIT - used
    require(total >= used and total + remaining <= 20 * 2**30 and free >= remaining,
            'project 20GiB/free-space budget exceeded')
    return dict(project_bytes=total, output_bytes=used, remaining_bytes=remaining)


def space(workspace, base):
    def du(path):
        return int(command('du', '-s', '-B1', str(path)).split()[0]) if path.exists() else 0
    return space_values(du(workspace), du(base), shutil.disk_usage(workspace).free)


def check_config(report, stage):
    cfg = report['cfg']
    require(all(cfg[k] is True for k in ('flag_relative_motion', 'flag_stitching', 'flag_normalize_lip',
            'flag_use_half_precision', 'flag_do_crop')), 'expected recorded on configuration')
    require(all(cfg[k] is False for k in ('flag_eye_retargeting', 'flag_lip_retargeting',
            'flag_source_video_eye_retargeting', 'flag_do_torch_compile', 'flag_crop_driving_video')) and
            cfg['driving_multiplier'] == 1 and cfg['animation_region'] == 'all' and
            cfg['driving_option'] == 'expression-friendly', 'unsupported ablation configuration')
    source = report['inputs_before'][report['ArgumentConfig']['source']]
    require((stage == 'student' and source == TWO_DRIVER_SOURCE) or
            (stage == 'off' and source in (OLD, NEW)), 'unapproved source/stage; new source is on/student only')
    return source


def verify_blobs(pin, paths, workspace, manifest, expected=None):
    require(re.fullmatch('[0-9a-f]{40}', pin), 'invalid reference code SHA')
    blobs = {}
    for rel in paths:
        path = tracked(ROOT / rel, workspace, manifest)
        blobs[rel] = command('git', 'hash-object', str(path))
        require(blobs[rel] == command('git', 'rev-parse', pin + ':' + rel) and
                (expected is None or expected[rel] == blobs[rel]), 'incompatible original blob: ' + rel)
    return blobs


def load_template(path, metadata):
    """The snapshot has 3488 numeric NPY entries, beyond bounded_npz's 64-entry cap."""
    require(metadata['n_frames'] == 581 and metadata['output_fps'] == 25, 'template length/fps')
    shapes = {'scale': (1, 1), 'R': (1, 3, 3), 'exp': (1, 21, 3),
              't': (1, 3), 'kp': (1, 21, 3), 'x_s': (1, 21, 3)}
    expected = {f'motion_{f:04d}_{k}': shape for f in range(581) for k, shape in shapes.items()}
    expected.update(c_eyes_lst=(581, 1, 2), c_lip_lst=(581, 1, 1))
    require(path.stat().st_size <= 4 * 2**20, 'template size limit')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {e.filename for e in entries} == {k + '.npy' for k in expected}
                and sum(e.file_size for e in entries) <= 4 * 2**20, 'template entries/expanded size')
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'unsupported NPY header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                require(shape == expected[entry.filename[:-4]] and dtype == np.float32 and
                        np.prod(shape) * 4 == entry.file_size - stream.tell(), 'unsafe template shape/dtype/payload')
    with np.load(path, allow_pickle=False) as values:
        result = dict(n_frames=581, output_fps=25, motion=[], c_eyes_lst=[], c_lip_lst=[])
        for f in range(581):
            result['motion'].append({k: values[f'motion_{f:04d}_{k}'].copy() for k in shapes})
            for k in ('c_eyes_lst', 'c_lip_lst'):
                result[k].append(values[k][f].copy())
    def check(value, meta):
        require(np.isfinite(value).all() and list(value.shape) == meta['shape'] and
                str(value.dtype) == meta['dtype'] and array_hash(value) == meta['array_sha256'], 'template array mismatch')
    for f in range(581):
        for k, value in result['motion'][f].items():
            check(value, metadata['motion'][f][k])
        for k in ('c_eyes_lst', 'c_lip_lst'):
            check(result[k][f], metadata[k][f])
    return result


def configured(cls, values):
    obj = cls()
    for k, value in values.items():
        if hasattr(obj, k):
            if value is None or isinstance(value, (bool, str, int, float)):
                setattr(obj, k, value)
            elif isinstance(value, dict) and 'array_sha256' in value:
                require(array_hash(getattr(obj, k)) == value['array_sha256'], 'config array changed: ' + k)
            else:
                require(list(getattr(obj, k)) == value, 'config sequence changed: ' + k)
    return obj


def motion_keys(report, arrays, crop, canvas, template, workspace, base, torch):
    from src import live_portrait_pipeline as pipeline
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.config.argument_config import ArgumentConfig
    # pipeline import enables benchmark globally; restore the verified snapshot policy.
    env = report['environment']
    torch.backends.cudnn.benchmark = env['cudnn_benchmark']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = env['tf32']
    torch.backends.cuda.matmul.allow_tf32 = env['tf32']
    torch.use_deterministic_algorithms(env['deterministic_algorithms'])
    cfg = configured(InferenceConfig, report['cfg'])
    cfg.flag_pasteback = False  # K-only collection; final pasteback uses original transform.
    wrapper = LivePortraitWrapper(cfg)
    models = {k: getattr(wrapper, k) for k in ('appearance_feature_extractor', 'motion_extractor',
              'warping_module', 'spade_generator')}
    models.update({'retarget:' + k: v for k, v in wrapper.stitching_retargeting_module.items()})
    for model in models.values():
        model.eval().requires_grad_(False)
    before = states(models)
    require(before == report['base_before'], 'pipeline base state mismatch')
    pipe = pipeline.LivePortraitPipeline.__new__(pipeline.LivePortraitPipeline)
    pipe.live_portrait_wrapper = wrapper
    pipe.cropper = SimpleNamespace(crop_cfg=configured(CropConfig, report['cropcfg']),
                                  crop_source_image=lambda *a, **kw: {k: v.copy() for k, v in crop.items()})
    # Preserve original prepare_source strides as well as the fixed cached values.
    source = wrapper.prepare_source(crop['img_crop_256x256'])
    require(array_hash(source.detach().cpu().numpy()) == report['snapshot_arrays']['source_input'], 'source input regression failed')
    fsource = torch.from_numpy(arrays['F']).cuda()
    source_info, result, checks = [], {}, []
    original_info, original_transform = wrapper.get_kp_info, wrapper.transform_keypoint
    def info(value, *a, **kw):
        require(torch.equal(value, source), 'source M input changed')
        if not source_info:
            source_info.append(original_info(source, *a, **kw))
        return {k: v.clone() for k, v in source_info[0].items()}
    def transform(value):
        xs = original_transform(value)
        require(array_hash(xs.detach().cpu().numpy()) == report['snapshot_arrays']['x_s'], 'source xs regression failed')
        return xs
    def prepare(image):
        require(np.array_equal(image, crop['img_crop_256x256']), 'source crop changed')
        return source.clone()
    class Complete(Exception):
        pass
    def denied(*a, **kw):
        raise RuntimeError('unexpected pipeline media write/detection path')
    for normalize in (True, False):
        keys = []
        cfg.flag_normalize_lip = normalize
        args = configured(ArgumentConfig, report['ArgumentConfig'])
        args.flag_normalize_lip = normalize
        def collect(f, xs, k):
            frame = len(keys)
            require(torch.equal(f, fsource) and array_hash(xs.detach().cpu().numpy()) == report['snapshot_arrays']['x_s'],
                    'source F/xs changed in motion collection')
            value = k.detach().cpu().numpy().copy()
            if normalize:
                require(np.array_equal(value, arrays['final_k'][frame]), f'on K regression failed {frame}')
            keys.append(value)
            if frame % 100 == 0:
                space(workspace, base)
            if frame == 580:
                raise Complete()
            return {'out': None}
        with ExitStack() as stack, torch.no_grad():
            # Route the existing template branch to the verified numeric cache, never load pickle.
            overrides = {'is_template': lambda p: p == args.driving,
                         'load': lambda p: copy.deepcopy(template), 'load_image_rgb': lambda p: canvas.copy(),
                         'dump': denied, 'mkdir': denied, 'images2video': denied}
            for name, value in overrides.items():
                stack.enter_context(patch.object(pipeline, name, value))
            for name, value in {'prepare_source': prepare, 'get_kp_info': info, 'transform_keypoint': transform,
                'extract_feature_3d': lambda value: fsource.clone(), 'warp_decode': collect,
                'parse_output': lambda value: [np.zeros((1, 1, 3), np.uint8)]}.items():
                stack.enter_context(patch.object(wrapper, name, value))
            try:
                pipe.execute(args)
            except Complete:
                pass
        require(len(keys) == 581, 'incomplete original pipeline K collection')
        result['on' if normalize else 'off'] = np.stack(keys)
        checks.append({'normalize_lip': normalize, 'frames': len(keys), 'final_k': array_hash(np.stack(keys))})
    require(states(models) == before, 'motion collection changed original model state')
    return result['off'], models, {'on_K_exact_all581': True, 'motion_runs': checks,
                                  'source_M_calls': len(source_info), 'base_before': before}


def load_student(path, workspace, snapshot, models, manifest, torch):
    from src.modules.real_teeth_decoder import RealTeethDecoder
    path = tracked(path, workspace, manifest, STUDENT)
    require(path.stat().st_size < 16 * 2**20, 'student size limit')
    owner = read_json(tracked(path.parent / 'report.json', workspace, manifest), 32 * 2**20)
    supervisor = read_json(tracked(path.parent / 'supervisor.json', workspace, manifest))
    meta_path = tracked(path.with_suffix('.json'), workspace, manifest,
                        owner['files'][path.with_suffix('.json').name]['sha256'])
    meta = read_json(meta_path, 16 * 2**20)
    require(owner['status'] == supervisor['status'] == 'completed' and owner['supervisor_verified'] is True and
            owner['stage'] == supervisor['stage'] == 'train' and owner['base_before'] == owner['base_after'] and
            owner['files'][path.name]['sha256'] == STUDENT == meta['checkpoint_sha256'], 'student owner/sha mismatch')
    paths = (*real.CORE, 'src/modules/real_teeth_decoder.py', 'scripts/run_real_teeth.py')
    require(set(meta['blobs']) == set(paths), 'student blob manifest mismatch')
    verify_blobs(meta['code_sha'], paths, workspace, manifest, meta['blobs'])
    require(owner['code_sha'] == meta['code_sha'] and meta['arch'] == real.ARCH and
            meta['threshold'] == .95 and meta['ROI'] == list(RealTeethDecoder.ROI) and
            meta['mask_steps'] == 600 and meta['student_steps'] == 300 and meta['checkpoint_reload_exact'] is True and
            all(meta['weights'].get(k) == v for k, v in snapshot['weights_before'].items()) and
            owner['base_before'] == {k: snapshot['base_before'][k] for k in owner['base_before']}, 'student compatibility failure')
    for name, expected in meta['weights'].items():
        tracked(name, workspace, manifest, expected)
    decoder = RealTeethDecoder(models['spade_generator'], threshold=.95).cuda()
    decoder.load_delta_state_dict(torch.load(path, weights_only=True, map_location='cuda'))
    decoder.set_training_stage('eval').eval()
    return decoder, {'checkpoint_sha256': STUDENT, 'training_code_sha': meta['code_sha'],
                     'metadata_sha256': sha256(meta_path), 'compatibility': 'unchanged original blobs; metadata not rewritten',
                     'strength': .5, 'training_quality_pass': meta['quality_pass']}


def render(args, workspace, snapshot, base, output, report, arrays, keys, crop, canvas, models, decoder, torch):
    import cv2
    from PIL import Image
    from src.config.inference_config import InferenceConfig
    from src.utils.crop import paste_back, prepare_paste_back
    cv2.setNumThreads(4)
    height, width = canvas.shape[:2]
    require(width % 2 == height % 2 == 0, 'even canvas required')
    template = InferenceConfig().mask_crop
    require(array_hash(template) == report['cfg']['mask_crop']['array_sha256'], 'pasteback mask changed')
    mask = prepare_paste_back(template, crop['M_c2o'], (width, height))
    driving = inside(report['ArgumentConfig']['driving'], workspace)
    audio = audio_signature(media(driving, audio=True))
    require(audio == report['audio_before'], 'source audio differs')
    fsource, xs = torch.from_numpy(arrays['F']).cuda(), torch.from_numpy(arrays['x_s']).cuda()
    w, g = models['warping_module'], models['spade_generator']
    rows, regressions = [], []
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '4', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '25', '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p', str(output / 'candidate_silent.mp4')],
        stdin=subprocess.PIPE)
    def original(k):
        with torch.autocast('cuda', dtype=torch.float16):
            warped = w(fsource, kp_driving=torch.from_numpy(k).cuda(), kp_source=xs)['out']
            if decoder is not None:
                return decoder.extract_base(warped)
            return g(warped)
    try:
        with torch.no_grad():
            for frame in range(581):
                if frame % 20 == 0:
                    space(workspace, base)
                base_raw = base_full = None
                if decoder is not None:
                    pre, seg, h, logits = original(keys[frame])
                    base_raw = tensor_image(logits.sigmoid())[1]
                    student = decoder.student_logits(pre, seg)
                    _, gate = decoder.predict_mask(decoder.mask_features(h))
                    require(torch.equal(decoder.compose(logits, student, gate, 0), logits.sigmoid()), 'student zero mismatch')
                    raw, support, audits = real.composed_audit(logits, decoder.compose(logits, student, gate, .5), gate)
                else:
                    raw = tensor_image(original(keys[frame]))[1]
                    audits = {}
                    if frame in SAMPLED:
                        base_raw = tensor_image(original(arrays['final_k'][frame]))[1]
                full = paste_back(raw, crop['M_c2o'], canvas, mask)
                if base_raw is not None:
                    require(array_hash(base_raw) == report['frames'][frame]['raw512'], f'on raw regression failed {frame}')
                    base_full = paste_back(base_raw, crop['M_c2o'], canvas, mask)
                    require(array_hash(base_full) == report['frames'][frame]['full_rgb'], f'on full regression failed {frame}')
                    regressions.append(frame)
                    if decoder is not None:
                        domain = cv2.warpAffine(support.astype(np.float32), crop['M_c2o'][:2], (width, height), flags=cv2.INTER_LINEAR) > 0
                        audits['full_propagation'] = pixel_audit(base_full, full, domain)
                        require(audits['full_propagation']['outside_max'] == 0, 'student pasteback leakage')
                    else:
                        audits['sampled_raw_change'] = pixel_audit(base_raw, raw, np.ones((512, 512), bool))
                encoder.stdin.write(np.ascontiguousarray(full).tobytes())
                rows.append({'frame': frame, 'K': array_hash(keys[frame]), 'K_changed': not np.array_equal(keys[frame], arrays['final_k'][frame]),
                             'raw512': array_hash(raw), 'full': array_hash(full), 'audits': audits,
                             'raw_changed': array_hash(raw) != report['frames'][frame]['raw512'],
                             'full_changed': array_hash(full) != report['frames'][frame]['full_rgb']})
                if frame in SAMPLED:
                    Image.fromarray(np.concatenate((base_raw, raw), axis=1)).save(output / f'raw_pair_f{frame:04d}.png')
        encoder.stdin.close()
        require(encoder.wait(timeout=60) == 0, 'encoder failed')
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    ffmpeg('-i', output / 'candidate_silent.mp4', '-i', driving, '-map', '0:v:0', '-map', '1:a:0',
           '-c', 'copy', '-copyts', '-avoid_negative_ts', 'disabled', output / 'candidate_full.mp4')
    filters = ';'.join(f'[{i}:v]scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]'
                      for i in range(2)) + ';[v0][v1]hstack=inputs=2[v]'
    ffmpeg('-i', snapshot / 'A0_full.mp4', '-i', output / 'candidate_full.mp4', '-i', driving,
           '-filter_complex_threads', '1', '-filter_complex', filters, '-map', '[v]', '-map', '2:a:0',
           '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
           '-copyts', '-avoid_negative_ts', 'disabled', output / 'compare_full.mp4')
    verification = {}
    for path in [snapshot / 'A0_full.mp4'] + [output / n for n in ('candidate_silent.mp4', 'candidate_full.mp4', 'compare_full.mp4')]:
        stream = check_video(media(path))
        ffmpeg('-xerror', '-i', path, '-f', 'null', '-')
        exact = None if path.name.endswith('_silent.mp4') else audio_signature(media(path, audio=True)) == audio
        require(exact is not False, 'audio packet hash/PTS/DTS mismatch')
        verification[path.name] = dict(stream=stream, full_decode_passed=True, audio_exact=exact)
    return dict(frames=rows, on_raw_regression_frames=regressions, video_verification=verification,
                audio_before=audio, audio_exact=True, nframes=581, fps=25, duration=23.24,
                baseline_video=str(snapshot / 'A0_full.mp4'), selected_frames=list(SAMPLED))


def worker(args, workspace, snapshot, base, output):
    manifest, result = {}, {}
    code = code_check()
    models = {}
    started = time.monotonic()
    try:
        report, arrays, manifest = prep.load_inputs(snapshot, workspace, manifest)
        source_sha = check_config(report, args.stage)
        require({'source_canvas.png', 'A0_full.mp4', 'driving_motion.npz'} <= set(report['files']), 'missing snapshot assets')
        verify_blobs(report['code_sha'], (*real.CORE, 'src/live_portrait_pipeline.py', 'src/live_portrait_wrapper.py',
            'src/modules/motion_extractor.py', 'src/modules/convnextv2.py', 'src/modules/stitching_retargeting_network.py',
            'src/utils/camera.py', 'src/utils/retargeting_utils.py', 'src/utils/helper.py',
            'src/utils/resources/lip_array.pkl', 'src/config/argument_config.py', 'src/config/crop_config.py'), workspace, manifest)
        crop = bounded_npz(snapshot / 'source_crop.npz', 16, set(report['crop_arrays']))
        from PIL import Image
        canvas = np.asarray(Image.open(snapshot / 'source_canvas.png').convert('RGB'))
        require(array_hash(canvas) == report['source_canvas']['array_sha256'], 'canvas hash differs')
        write_json(output / 'report.json', {'status': 'running', 'input_hashes': manifest, 'supervisor_verified': False})
        torch = runtime(report)
        gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        decoder, provenance = None, {}
        if args.stage == 'off':
            template = load_template(snapshot / 'driving_motion.npz', report['template'])
            keys, models, provenance = motion_keys(report, arrays, crop, canvas, template, workspace, base, torch)
        else:
            models = load_base(workspace, report, torch)
            keys = arrays['final_k'].copy()
            decoder, provenance = load_student(inside(args.checkpoint, workspace), workspace, report, models, manifest, torch)
        before = states(models)
        require(before == {k: report['base_before'][k] for k in models}, 'loaded original state mismatch')
        student_before = None if decoder is None else {k: array_hash(v.cpu().numpy()) for k, v in decoder.delta_state_dict().items()}
        write_json(output / 'report.json', {'status': 'running', 'input_hashes': manifest, 'base_before': before,
                                           'provenance': provenance, 'supervisor_verified': False})
        result = render(args, workspace, snapshot, base, output, report, arrays, keys, crop, canvas, models, decoder, torch)
        gpu_end.record()
        torch.cuda.synchronize()
        after = states(models)
        require(before == after and code_check() == code and all(sha256(p) == h for p, h in manifest.items()), 'input/model/code mutation')
        require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in models.values()), 'base unfrozen')
        if decoder is not None:
            require(student_before == {k: array_hash(v.cpu().numpy()) for k, v in decoder.delta_state_dict().items()}, 'student mutation')
        require(prep.validate_snapshot(arrays) == report['snapshot_arrays'], 'source snapshot arrays changed')
        np.savez(output / 'final_k.npz', final_k=keys)
        result.update(status='completed', stage=args.stage, quality_pass=False, code_sha=code, source_sha256=source_sha,
            intervention='normalize_lip False; original pipeline K rebuilt' if decoder is None else 'fixed historical decoder strength0.5; on K unchanged',
            provenance=provenance, source_snapshot_hashes=report['snapshot_arrays'], base_before=before, base_after=after,
            input_hashes=manifest, weights_before=report['weights_before'], weights_after={p: sha256(p) for p in report['weights_before']},
            labels_read=False, GT_prediction_input=False, source_redetected=False, source_F_recomputed=False,
            semantic_safety='not established; full video requires parent independent review',
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), environment=report['environment'],
            cuda_timeline_seconds=gpu_start.elapsed_time(gpu_end) / 1000.,
            worker_wall_seconds=time.monotonic() - started, files=real.inventory(output), supervisor_verified=False)
        write_json(output / 'report.json', result)
    except BaseException as exc:
        result.update(status='failed', stage=args.stage, quality_pass=False, error=repr(exc), input_hashes=manifest,
                      worker_wall_seconds=time.monotonic() - started, supervisor_verified=False)
        write_json(output / 'report.json', result)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('off', 'student'), required=True)
    for name in ('workspace', 'snapshot', 'budget-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--authorize-source-mouth', action='store_true')
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main():
    started = time.monotonic()
    args = parser().parse_args()
    require((args.stage == 'student') == bool(args.checkpoint), 'checkpoint required only for student')
    require(sys.platform == 'linux' and args.authorize_source_mouth, 'explicit authorized Linux execution required')
    workspace, snapshot, base, output = prep.check_paths(args)
    for key in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)) and inside(os.environ[key], workspace).is_dir(), 'missing isolated ' + key)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('IMAGEIO_FFMPEG_NO_PREVENT_SIGINT') == '1', 'bytecode/ImageIO process group isolation required')
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'one explicit GPU UUID required')
    code_check()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if args._worker:
        require(os.environ.get('SOURCE_MOUTH_PARENT') == str(os.getppid()), 'supervised worker only')
        worker(args, workspace, snapshot, base, output)
        return
    require(not output.exists(), 'output must be new')
    import fcntl
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'source-mouth.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = elapsed_charge(base)
        require(1800 - prior >= 600, 'insufficient cumulative 1800s allocation')
        for path in base.rglob('supervisor.json'):
            old = read_json(path)
            require(not (old.get('status') == 'completed' and old.get('stage') == args.stage and
                         old.get('snapshot') == str(snapshot)), 'condition already completed; no extra trial')
        initial = space(workspace, base)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'supervisor.json', {'status': 'running', 'stage': args.stage, 'snapshot': str(snapshot)})
        owned = timer = process = None
        status, error = 'failed', None
        try:
            env = dict(os.environ, SOURCE_MOUTH_PARENT=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[key] = '4'
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                                       cwd=ROOT, env=env, start_new_session=True)
            owned = OwnedProcessGroup(process)
            timer = threading.Timer(max(0, 585 - (time.monotonic() - started)), owned.kill)
            timer.daemon = True
            timer.start()
            while not owned.exited():
                require(time.monotonic() - started < 585, 'worker deadline; reserve final audit time')
                space(workspace, base)
                time.sleep(.5)
            owned.finish()
            require(process.returncode == 0, 'worker failed')
            result = read_json(output / 'report.json', 32 * 2**20)
            require(result['status'] == 'completed', 'incomplete worker report')
            result.update(budget_before=initial, budget_after=space(workspace, base), supervisor_verified=True)
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
            result = read_json(output / 'report.json', 32 * 2**20) if (output / 'report.json').exists() else {}
            after = {}
            for p in result.get('input_hashes', {}):
                try:
                    after[p] = sha256(p)
                except OSError as exc:
                    after[p] = repr(exc)
            wall = time.monotonic() - started
            if wall > 600 or prior + wall > 1800 or after != result.get('input_hashes', {}):
                status, error = 'failed', error or 'final wall/input audit failed'
            record = dict(status=status, error=error, stage=args.stage, snapshot=str(snapshot), wall_seconds=wall,
                          cumulative_wall_seconds=prior + wall, hard_limit_seconds=600,
                          returncode=None if process is None else process.returncode)
            result.update(status=status, supervisor_verified=status == 'completed', input_hashes_after=after, supervisor=record)
            write_json(output / 'report.json', result)
            write_json(output / 'supervisor.json', record)
        require(status == 'completed', 'supervisor rejected run')


if __name__ == '__main__':
    main()
