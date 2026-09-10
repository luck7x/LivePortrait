"""Conservative, aligned-patch RGB gap filling implemented only with NumPy."""
import numpy as np


def _shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    if abs(dy) < h and abs(dx) < w:
        out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = a[
            max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def _coherent_gaps(mask, scale):
    """Keep vertically supported holes; reject isolated dots and thin bridges."""
    keep = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points = []
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = np.array(points).T
        height, width = int(ys.max()-ys.min()+1), int(xs.max()-xs.min()+1)
        if (height >= max(2, int(np.ceil(1.5 * scale)))
                and width <= 1.5 * height and len(points) >= max(2, int(np.ceil(scale * scale)))):
            keep[ys, xs] = True
    return keep


def transition_weights(active):
    """Bounded offline two-frame ramp; rejected frames and one-frame islands stay off."""
    active = np.asarray(active)
    if active.ndim != 1 or active.dtype != np.bool_:
        raise ValueError('active must be a one-dimensional boolean sequence')
    weights = np.zeros(len(active), np.float32)
    for i in np.flatnonzero(active):
        lo, hi = int(i), int(i)
        while lo > 0 and active[lo-1]: lo -= 1
        while hi+1 < len(active) and active[hi+1]: hi += 1
        if hi > lo:
            weights[i] = min(1.0, (i-lo+1)/2.0, (hi-i+1)/2.0)
    return weights


def cleanup(rgb, bbox, strength=0.85, previous=None):
    """Return (RGB, current allowed bool mask, alpha state or None, reason).

    bbox is (x, y, width, height) in the already aligned context patch.
    State is anchor-coordinate alpha, never old RGB. A rejection clears state.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB must be HxWx3 uint8')
    h, w = rgb.shape[:2]
    if len(bbox) != 4 or any(not isinstance(v, (int, np.integer)) for v in bbox):
        raise ValueError('bbox must contain four integers')
    x, y, bw, bh = bbox
    if min(h, w, bw, bh) <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
        raise ValueError('bbox outside patch')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite in [0, 1]')
    if previous is not None and (not isinstance(previous, np.ndarray) or previous.shape != (h, w)
                                or not np.issubdtype(previous.dtype, np.floating)
                                or not np.isfinite(previous).all() or (previous < 0).any() or (previous > 1).any()):
        raise ValueError('invalid alpha state')
    empty = np.zeros((h, w), dtype=bool)
    def reject(reason):
        return rgb.copy(), empty, None, reason
    if strength == 0:
        return reject('zero-strength')
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    # Reject red lips / warm skin as well as dim mouth pixels. This is a
    # photometric heuristic, not anatomical segmentation.
    red = (r - g > 45) | (r - b > 65)
    white = (g >= 105) & (b >= 85) & (r - g <= 65) & ~red
    counts = white[:, x:x + bw].sum(axis=1)
    rows = counts >= max(4, int(np.ceil(bw * 0.28)))
    edges = np.diff(np.r_[False, rows, False].astype(np.int8))
    bands = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    scale = bw / 75.0
    gap = max(2, int(round(2 * scale)))
    expected = y + (bh - 1) / 2
    candidates = [(a, z) for a, z in bands if abs((a + z - 1) / 2 - expected) <= max(2, bh / 2)
                  and a >= y and z <= y + bh]
    if len(candidates) != 1:
        return reject('upper-band-ambiguous')
    a, z = candidates[0]
    lower = [(c, d) for c, d in bands if c >= z + gap]
    if not lower:
        return reject('no-separated-lower-band')
    c, _ = lower[0]
    # Require a genuinely dark intervening run, not merely fewer white teeth.
    dark = ((g[z:c, x:x + bw] < 100) & (b[z:c, x:x + bw] < 85)).mean(axis=1) >= 0.7
    if not any(dark[i:i + gap].all() for i in range(len(dark) - gap + 1)):
        return reject('no-dark-inter-row-gap')
    seed = empty.copy()
    seed[a:z, x:x + bw] = white[a:z, x:x + bw]
    domain = empty.copy()
    for row in range(a, z):
        xs = np.flatnonzero(seed[row])
        if len(xs) >= 2:
            domain[row, xs[0]:xs[-1] + 1] = True
    kernel = min(15, max(3, int(round(5 * scale)) | 1))
    radius = kernel // 2
    dilated = np.logical_or.reduce([_shift(seed, 0, dx) for dx in range(-radius, radius + 1)])
    closed = np.logical_and.reduce([_shift(dilated, 0, dx) for dx in range(-radius, radius + 1)])
    fill = closed & domain & ~seed & ~red
    margin = max(1, int(round(0.5 * scale)))
    fill[:a + margin] = False
    fill[max(a, z - margin):] = False
    fill = _coherent_gaps(fill, scale)
    if not fill.any():
        return reject('no-coherent-vertical-gaps')
    radius = min(7, max(2, int(round(2 * scale))))
    total = np.zeros_like(rgb, dtype=np.float32)
    weights = np.zeros((h, w), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = 1.0 / (1 + dx * dx + dy * dy)
            support = _shift(seed, dy, dx) * weight
            weights += support
            total += _shift(rgb.astype(np.float32), dy, dx) * support[..., None]
    fill &= weights > 0
    alpha = fill.astype(np.float32) * strength
    if previous is not None:
        alpha = (0.65 * alpha + 0.35 * previous) * fill
    target = total / np.maximum(weights[..., None], 1e-8)
    out = rgb.copy()
    mixed = np.rint(rgb * (1 - alpha[..., None]) + target * alpha[..., None]).clip(0, 255).astype(np.uint8)
    out[fill] = mixed[fill]
    assert np.array_equal(out[~fill], rgb[~fill])
    return out, fill, alpha, 'ok'
