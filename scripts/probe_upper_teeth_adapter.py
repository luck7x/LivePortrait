"""Linux-only bounded connectivity smoke test. Never a customer-quality training run."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

BASE_SHA = '31c26a497cedf336a813e18b61e96239f7cab878'
ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def inside(path, workspace, exists=True):
    path = Path(path).resolve(strict=exists)
    require(path.is_relative_to(workspace), 'path escapes workspace: ' + str(path))
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=20).strip()


def budget(workspace, output):
    total = int(command('du', '-s', '-B1', str(workspace)).split()[0])
    used = int(command('du', '-s', '-B1', str(output)).split()[0]) if output.exists() else 0
    remaining = max(0, 128*2**20-used)
    require(total + remaining <= 20*2**30 and used <= 128*2**20, 'allocated-space budget/headroom exceeded')
    require(shutil.disk_usage(workspace).free >= remaining, 'insufficient filesystem headroom')
    return {'workspace_allocated_bytes': total, 'output_allocated_bytes': used}


def code_check():
    sha = command('git', 'rev-parse', 'HEAD')
    require(len(os.environ.get('PROBE_CODE_SHA', '')) == 40 and sha == os.environ['PROBE_CODE_SHA'], 'PROBE_CODE_SHA mismatch')
    require(not command('git', 'status', '--porcelain', '--untracked-files=all'), 'dirty Git tree')
    return sha


def worker(args, workspace, output):
    # Imports occur only after Linux/auth/path guards and inside the supervised child.
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(ROOT))
    from src.modules.spade_generator import SPADEDecoder
    from src.modules.warping_network import WarpingNetwork
    from src.modules.upper_teeth_adapter import UpperTeethAdapterDecoder, safe_feature_mask
    from src.utils.upper_teeth_support import safe_feature_mask as numpy_mask

    require(torch.cuda.device_count() == 1, 'exactly one visible GPU required')
    torch.manual_seed(20260911)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    code = code_check()
    files = [inside(p, workspace) for p in (args.feature_npz, args.keypoints_npz, args.provenance_json)]
    input_hashes = {str(p): sha256(p) for p in files}
    provenance = json.loads(files[2].read_text())
    require(provenance['status'] == 'completed' and provenance['postflight_verified'], 'unverified causal provenance')
    entries = [r for r in provenance['cases']['baseline']['frames'] if r['frame'] == 263]
    require(len(entries) == 1, 'need unique baseline f263 provenance')
    entry = entries[0]
    weights = {inside(p, workspace): digest for p, digest in provenance['weights_before'].items()}
    require(len(weights) == 5 and provenance['weights_before'] == provenance['weights_after'], 'need five unchanged weight hashes')
    require(all(sha256(p) == h for p, h in weights.items()), 'weight provenance mismatch')

    def array_hash(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

    arrays = {}
    for path, keys in [(files[0], ('f_s', 'source_input')), (files[1], ('x_s', 'final_k'))]:
        require(path.stat().st_size <= 128*2**20, 'oversized NPZ')
        # Refuse compressed bombs before materializing, and never deserialize objects.
        import zipfile
        with zipfile.ZipFile(path) as z:
            require(sum(i.file_size for i in z.infolist()) <= 128*2**20, 'oversized expanded NPZ')
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == set(keys), 'unexpected NPZ keys')
            arrays.update({k: z[k] for k in keys})
    expected = {'f_s': (1, 32, 16, 64, 64), 'source_input': (1, 3, 256, 256),
                'x_s': (1, 21, 3), 'final_k': (1, 21, 3)}
    for k, a in arrays.items():
        require(a.shape == expected[k] and a.dtype == np.float32 and np.isfinite(a).all(), 'invalid numeric input ' + k)
        key = {'source_input': 'prepare_source', 'final_k': 'final_k_hash'}.get(k, k)
        require(array_hash(a) == entry[key], 'array provenance mismatch ' + k)
    cfg = yaml.safe_load(inside(ROOT / 'src/config/models.yaml', workspace).read_text())['model_params']

    def load(model, filename):
        paths = [p for p in weights if p.name == filename]
        require(len(paths) == 1, 'missing/ambiguous weight ' + filename)
        state = torch.load(paths[0], map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.cuda().eval().requires_grad_(False)

    g = load(SPADEDecoder(**cfg['spade_generator_params']), 'spade_generator.pth')
    w = load(WarpingNetwork(**cfg['warping_module_params']), 'warping_module.pth')
    # Trusted immutable repository source, not an external pickle or user code string.
    old_source = command('git', 'show', BASE_SHA + ':src/modules/spade_generator.py')
    namespace = {'__name__': 'src.modules._trusted_original_spade', '__package__': 'src.modules'}
    exec(compile(old_source, BASE_SHA + ':spade_generator.py', 'exec'), namespace)
    original = load(namespace['SPADEDecoder'](**cfg['spade_generator_params']), 'spade_generator.pth')

    def state_hash(model):
        return {**{'state:' + k: array_hash(v.detach().cpu().numpy()) for k, v in model.state_dict().items()},
                **{'buffer:' + k: array_hash(v.detach().cpu().numpy()) for k, v in model.named_buffers()}}

    frozen = {'g': state_hash(g), 'w': state_hash(w), 'original': state_hash(original)}
    require(frozen['g'] == frozen['original'], 'original state_dict mismatch')
    tensors = {k: torch.from_numpy(a).cuda() for k, a in arrays.items() if k != 'source_input'}
    with torch.no_grad():
        feature = w(tensors['f_s'], kp_source=tensors['x_s'], kp_driving=tensors['final_k'])['out']
    model = UpperTeethAdapterDecoder(g).cuda()
    structure = torch.zeros((1, 3, 256, 256), device='cuda')
    structure[:, 0] = .7
    structure[:, 1] = torch.linspace(0, 1, 256, device='cuda').view(1, 256, 1)
    structure[:, 2] = .2
    allowed = torch.zeros((1, 1, 512, 512), dtype=torch.bool, device='cuda')
    allowed[..., 300:348, 180:332] = True  # SYNTHETIC rectangle, not a tooth annotation.
    visibility = allowed.float()
    require(np.array_equal(safe_feature_mask(allowed, visibility).cpu().numpy(),
                           numpy_mask(allowed.cpu().numpy(), visibility.cpu().numpy())), 'support oracle mismatch')

    def uint8(x):
        return (x.detach().float().clamp(0, 1)*255).clamp(0, 255).to(torch.uint8)

    def exact(a, b, label):
        require(torch.equal(a, b) and torch.equal(uint8(a), uint8(b)), label + ' not exact')

    checks = {}
    for half in (False, True):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=half):
            reference = original(feature)
            exact(reference, g(feature), 'split G')
            exact(reference, model(feature), 'unconditional')
            exact(reference, model(feature, structure, visibility, allowed), 'zero init')
        checks['fp16' if half else 'fp32'] = 'tensor and uint8 exact'
    with torch.no_grad():
        baseline = g(feature)
    target = (baseline + .01*visibility).clamp(0, 1)  # Synthetic numerical objective, NOT GT.
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=.005)
    model.train()
    gradients = []
    for _ in range(3):
        budget(workspace, output)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(feature, structure, visibility, allowed)
        loss = ((prediction-target).square()*visibility).sum() / (visibility.sum()*3)
        loss.backward()
        grads = [p.grad for p in model.adapter.parameters()]
        require(all(v is not None and torch.isfinite(v).all().item() for v in grads), 'nonfinite/missing gradient')
        norm = sum(v.abs().sum().item() for v in grads)
        require(norm > 0, 'zero adapter gradient')
        gradients.append({'loss': loss.item(), 'gradient_l1': norm})
        optimizer.step()
    with torch.no_grad():
        prediction = model(feature, structure, visibility, allowed)
        diff = (prediction-baseline).abs()
        outside = (~allowed).expand_as(diff)
        float_out = diff[outside].max().item()
        uint_out = (uint8(prediction).int()-uint8(baseline).int()).abs()[outside].max().item()
        partial = {'stage': 'synthetic_update_before_acceptance', 'accepted': False,
                   'synthetic': True, 'steps': gradients, 'checks': checks,
                   'outside_float_max': float_out, 'outside_uint8_max': uint_out,
                   'actual_max_float_change': diff.max().item()}
        (output / 'probe-partial.json').write_text(json.dumps(partial, indent=2))
        require(torch.isfinite(prediction).all().item() and diff.max().item() > 0, 'no finite actual change')
        require(float_out == 0 and uint_out == 0, 'outside support leakage')
        exact(model(feature, structure, visibility*0, allowed), baseline, 'vis0 fallback')
        exact(model(feature, structure, visibility, allowed & False), baseline, 'empty fallback')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        baseline_half = g(feature)
        prediction_half = model(feature, structure, visibility, allowed)
        half_diff = (prediction_half-baseline_half).abs()
        half_out = half_diff[outside].max().item()
        half_uint_out = (uint8(prediction_half).int()-uint8(baseline_half).int()).abs()[outside].max().item()
        checks['fp16_after_update'] = {'outside_float_max': half_out, 'outside_uint8_max': half_uint_out}
        partial['checks'] = checks
        (output / 'probe-partial.json').write_text(json.dumps(partial, indent=2))
        require(half_out == 0 and half_uint_out == 0, 'FP16 outside support leakage')
    require(frozen == {'g': state_hash(g), 'w': state_hash(w), 'original': state_hash(original)}, 'base state/buffer mutation')
    checkpoint = {'adapter': model.adapter.state_dict(), 'base_weight_hashes': {str(p): h for p, h in weights.items()},
                  'code_sha': code, 'architecture': {'width': model.width, 'max_delta': model.max_delta}, 'synthetic': True}
    torch.save(checkpoint, output / 'synthetic-smoke.pt')
    restored = torch.load(output / 'synthetic-smoke.pt', map_location='cuda', weights_only=True)
    require(restored['synthetic'] is True and restored['code_sha'] == code
            and restored['base_weight_hashes'] == checkpoint['base_weight_hashes'], 'checkpoint metadata mismatch')
    clone = UpperTeethAdapterDecoder(g, **restored['architecture']).cuda().eval()
    clone.adapter.load_state_dict(restored['adapter'], strict=True)
    with torch.no_grad():
        exact(prediction, clone(feature, structure, visibility, allowed), 'weights_only reload')
    require(frozen == {'g': state_hash(g), 'w': state_hash(w), 'original': state_hash(original)}, 'post-reload base/buffer mutation')
    require(all(sha256(p) == h for p, h in weights.items()), 'weight files changed')
    require(all(sha256(p) == h for p, h in input_hashes.items()), 'input files changed')
    require(code_check() == code, 'code changed')
    report = {'status': 'success', 'kind': 'model-internal connectivity only',
              'structure_predictor': 'not implemented', 'visual_approval': False, 'synthetic': True,
              'objective': 'baseline + 0.01 inside synthetic rectangle; not tooth GT',
              'code_sha': code, 'original_code_sha': BASE_SHA, 'original_source_sha256': hashlib.sha256(old_source.encode()).hexdigest(),
              'input_hashes': input_hashes, 'weights': {str(p): h for p, h in weights.items()},
              'checks': checks, 'steps': gradients, 'actual_max_float_change': diff.max().item(),
              'actual_max_uint8_change': (uint8(prediction).int()-uint8(baseline).int()).abs().max().item(),
              'outside_float_max': float_out, 'outside_uint8_max': uint_out,
              'base_parameters_and_all_buffers_unchanged': True, 'reload_exact': True,
              'checkpoint_sha256': sha256(output / 'synthetic-smoke.pt'),
              'adapter_parameters': sum(p.numel() for p in model.adapter.parameters()),
              'synthetic_mask_pixels': allowed.sum().item(), 'gpu': torch.cuda.get_device_name(0),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              **budget(workspace, output)}
    (output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'feature-npz', 'keypoints-npz', 'provenance-json', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--authorize-experimental', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and args.authorize_experimental, 'Linux and explicit experimental authorization required')
    workspace = Path(args.workspace).resolve(strict=True)
    require(workspace.is_dir(), 'workspace must be an existing authorized project directory')
    inside(ROOT, workspace)
    for path in (args.feature_npz, args.keypoints_npz, args.provenance_json):
        inside(path, workspace)
    output = inside(args.output, workspace, exists=False)
    require(not output.is_relative_to(ROOT), 'output must be outside code worktree')
    # All writable library caches/temp locations must already exist inside workspace.
    for key in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME', 'CUDA_CACHE_PATH'):
        require(bool(os.environ.get(key)), 'missing isolated cache ' + key)
        inside(os.environ[key], workspace)
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'disable bytecode writes')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'set deterministic CUBLAS_WORKSPACE_CONFIG')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(visible) and ',' not in visible and visible != '-1', 'select exactly one GPU explicitly')
    code_check()
    if args._worker:
        require(os.environ.get('UPPER_ADAPTER_SUPERVISOR_PID') == str(os.getppid()), 'worker must be supervised')
        worker(args, workspace, output)
        return
    require(not output.exists(), 'refuse existing output directory')
    budget(workspace, output)
    output.mkdir(parents=True)
    env = dict(os.environ, UPPER_ADAPTER_SUPERVISOR_PID=str(os.getpid()))
    require(time.monotonic()-started < 300, 'preflight deadline exceeded')
    process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                               cwd=ROOT, env=env, start_new_session=True)
    def kill_owned_group():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    timer = threading.Timer(max(0, 300 - (time.monotonic()-started)), kill_owned_group)
    timer.daemon = True
    timer.start()
    try:
        while process.poll() is None:
            require(time.monotonic()-started < 300, '300s hard deadline')
            budget(workspace, output)
            time.sleep(.5)
        require(process.returncode == 0, 'worker failed; not a successful probe')
        budget(workspace, output)
    finally:
        timer.cancel()
        # Only terminate our own still-running child; do not signal a stale group ID.
        kill_owned_group()
        process.wait()


if __name__ == '__main__':
    main()
