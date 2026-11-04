"""Step-by-step HTML debug report of the pipeline on a rosbag (offline).

    ros2 run thermal_change_detection debug_report <bag> -o out/ \\
        --params config/params.yaml --image /camera_front_thermo/image_raw/compressed \\
        --camera-info /camera_front_thermo/camera_info --map-frame FP_ENU0

Runs exactly the same pipeline as the node / offline_eval (same subsampling,
same config) with ``debug=True`` and writes

    out/index.html          summary: status timeline, top-down map, metric
                            charts, table of all processed frames
    out/frames/fNNNNN.html  one page per frame: pose change to the keyframe,
                            pose-prior homography, ECC per pyramid level,
                            quality gate, every change-detection step, blobs

Open out/index.html in a browser (no server needed). Use --detail to limit
the per-frame pages (and images) to a range of frame indices, e.g. 50:80.
"""
import argparse
import html
import math
import os
import shutil

import cv2
import numpy as np

from .core.align import _centre_shift, masked_correlation, preprocess, warp_to_current
from .core.config import PipelineConfig, load_ros_params_yaml
from .core.geometry import optical_axis
from .core.pipeline import ChangeDetectionPipeline
from .offline_eval import add_bag_args, iter_frames, load_tf

# Status categories: (label, colour). Colours follow a fixed status palette;
# every use is accompanied by a text label.
CATS = {
    'accepted': ('accepted', '#0ca30c'),
    'rej_shift': ('rejected: correction', '#ec835a'),
    'rej_corr': ('rejected: correlation', '#d03b3b'),
    'rej_overlap': ('rejected: overlap', '#fab219'),
    'noref': ('no reference', '#898781'),
}
# Overlay colours on images (BGR): pose prior / ECC result.
C_PRIOR = (0, 190, 255)     # yellow
C_ECC = (255, 200, 40)      # cyan
JPG = [cv2.IMWRITE_JPEG_QUALITY, 90]


# --------------------------------------------------------------------------- images

def norm8(img, mask=None, lo_pct=1.0, hi_pct=99.0, lims=None):
    """float image -> uint8 with percentile stretch over mask; returns (img8, (lo, hi))."""
    if lims is None:
        v = img[mask > 0] if mask is not None else img.ravel()
        v = v[np.isfinite(v)]
        lo, hi = (np.percentile(v, [lo_pct, hi_pct]) if v.size else (0.0, 1.0))
        lims = (float(lo), float(max(hi, lo + 1e-6)))
    lo, hi = lims
    f = (np.nan_to_num(img.astype(np.float32), nan=lo) - lo) * (255.0 / (hi - lo))
    return np.clip(f, 0, 255).astype(np.uint8), lims


def hatch_invalid(bgr, valid):
    """Darken + hatch pixels outside ``valid`` (same look as the node overlay)."""
    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inv = ~valid.astype(bool)
    out = bgr.copy()
    out[inv] = (out[inv] * 0.35 + 30).astype(np.uint8)
    out[inv & (((xx + yy) % 12) < 2)] = 100
    return out


def gray_bgr(g8):
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)


def blend_mg(cur8, ref8, mask):
    """False-colour overlay: current = green, reference = magenta, equal = gray."""
    out = np.stack([ref8, cur8, ref8], axis=-1)
    m = mask.astype(bool)
    out[~m] = (gray_bgr(cur8)[~m] * 0.45).astype(np.uint8)
    return out


def _lut_diverging():
    """Signed value in [-1, 1] -> BGR: blue - neutral gray - red."""
    blue = np.array([214, 120, 42], np.float32)    # #2a78d6 (BGR)
    mid = np.array([128, 128, 128], np.float32)
    red = np.array([72, 73, 227], np.float32)      # #e34948 (BGR)
    t = np.linspace(-1, 1, 256)[:, None]
    lut = np.where(t < 0, mid + (blue - mid) * (-t), mid + (red - mid) * t)
    return lut.astype(np.uint8)


LUT_DIV = _lut_diverging()


def diverging(v, vmax, valid):
    idx = np.clip((np.nan_to_num(v) / vmax + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return hatch_invalid(LUT_DIV[idx], valid)


def z_classes(z, valid, z_low, z_high, base8):
    """Gray image, light tint where z_low <= |z| < z_high, strong tint above."""
    out = gray_bgr((base8 * 0.6).astype(np.uint8)).astype(np.float32)
    zz = np.nan_to_num(z)
    for sign, col in ((1, (72, 73, 227)), (-1, (214, 120, 42))):
        weak = valid & (sign * zz >= z_low) & (sign * zz < z_high)
        strong = valid & (sign * zz >= z_high)
        out[weak] = out[weak] * 0.45 + np.array(col) * 0.55 * 0.6 + 40
        out[strong] = np.array(col)
    return hatch_invalid(np.clip(out, 0, 255).astype(np.uint8), valid)


def mask_image(base8, valid, layers):
    """layers: list of (bool mask, BGR) painted over a dimmed base image."""
    out = gray_bgr((base8 * 0.5).astype(np.uint8))
    for m, col in layers:
        out[m] = col
    return hatch_invalid(out, valid)


def _proj(H, pts):
    p = H @ np.vstack([pts.T, np.ones(len(pts))])
    ok = p[2] > 1e-9
    xy = p[:2] / np.where(ok, p[2], 1.0)
    return xy.T, ok


def draw_quad(img, H, w, h, off, color, label=None, thick=2):
    """Draw the border of a (w x h) image mapped by H, shifted by ``off`` px."""
    n = 40
    edge = np.concatenate([
        np.stack([np.linspace(0, w - 1, n), np.zeros(n)], 1),
        np.stack([np.full(n, w - 1), np.linspace(0, h - 1, n)], 1),
        np.stack([np.linspace(w - 1, 0, n), np.full(n, h - 1)], 1),
        np.stack([np.zeros(n), np.linspace(h - 1, 0, n)], 1)])
    xy, ok = _proj(H, edge)
    xy = np.clip(xy + off, -20000, 20000)
    pts = [tuple(int(round(c)) for c in p) for p in xy]
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        if ok[i] and ok[j]:
            cv2.line(img, pts[i], pts[j], color, thick, cv2.LINE_AA)
    if label and ok[0]:
        cv2.putText(img, label, (pts[0][0] + 4, pts[0][1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


def footprint_canvas(base8, W_list, w, h, pad=200):
    """Base image padded by ``pad`` px with footprints (H, colour, label) drawn on it."""
    canvas = np.full((base8.shape[0] + 2 * pad, base8.shape[1] + 2 * pad, 3), 35, np.uint8)
    canvas[pad:pad + base8.shape[0], pad:pad + base8.shape[1]] = gray_bgr(base8)
    cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + base8.shape[1], pad + base8.shape[0]),
                  (120, 120, 120), 1)
    for H, col, label in W_list:
        draw_quad(canvas, H, w, h, pad, col, label)
    return canvas


def corner_arrows(canvas, W0, W, w, h, pad):
    """Arrows from the prior to the ECC position of the 4 current-image corners
    and the centre (in reference px); the centre one is what the gate measures."""
    c = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                  [(w - 1) / 2, (h - 1) / 2]], float)
    p0, ok0 = _proj(W0, c)
    p1, ok1 = _proj(W, c)
    for a, b, o0, o1 in zip(p0, p1, ok0, ok1):
        if not (o0 and o1):
            continue
        a = tuple(int(v) for v in np.clip(a + pad, -20000, 20000))
        b = tuple(int(v) for v in np.clip(b + pad, -20000, 20000))
        cv2.arrowedLine(canvas, a, b, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.08)
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        cv2.putText(canvas, f'{d:.0f}px', (b[0] + 4, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)


def gain_density(x, y, a, b, size=300):
    """2-D histogram of (reference, current) values with the fitted line."""
    lo = float(np.percentile(np.concatenate([x, y]), 0.5))
    hi = float(np.percentile(np.concatenate([x, y]), 99.5))
    hi = max(hi, lo + 1.0)
    hist, _, _ = np.histogram2d(y, x, bins=size, range=[[lo, hi], [lo, hi]])
    v = np.log1p(hist)
    v = (v / max(v.max(), 1e-9))[::-1]          # y up
    col = np.array([229, 135, 57], np.float32)  # #3987e5 BGR
    img = (25 + v[..., None] * (col - 25)).astype(np.uint8)
    img[hist[::-1] == 0] = 25

    def px(xv, yv):
        return (int((xv - lo) / (hi - lo) * (size - 1)),
                int((1 - (yv - lo) / (hi - lo)) * (size - 1)))
    cv2.line(img, px(lo, lo), px(hi, hi), (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(img, px(lo, a * lo + b), px(hi, a * hi + b), (72, 73, 227), 2, cv2.LINE_AA)
    return img, (lo, hi)


def save(out_dir, name, img, png=False):
    path = os.path.join(out_dir, name)
    if png:
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, img, JPG)
    return name


# --------------------------------------------------------------------------- geometry

def pose_change(T_kf, T_cur, K):
    """Relative pose of the current camera in the keyframe camera frame."""
    T_rel = np.linalg.inv(T_kf) @ T_cur
    t = T_rel[:3, 3]
    R = T_rel[:3, :3]
    f = R[:, 2]
    tilt = math.degrees(math.asin(float(np.clip(-f[1], -1, 1))))   # + = up (y is down)
    pan = math.degrees(math.atan2(f[0], f[2]))                      # + = right
    a, b = math.radians(pan), math.radians(tilt)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
    M = (Ry @ Rx).T @ R
    roll = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    rvec, _ = cv2.Rodrigues(R)
    fx = float(K[0, 0])
    r_corner = math.hypot(K[0, 2], K[1, 2])
    lateral = math.hypot(t[0], t[1])
    # Pixel error of the rotation-only homography for a scene at depth Z
    # (lateral parallax + scaling from forward motion, at the image corner).
    parallax = {Z: fx * lateral / Z + r_corner * abs(t[2]) / max(Z - t[2], 1e-3)
                for Z in (2.0, 5.0, 10.0, 20.0)}
    return dict(t=t, dist=float(np.linalg.norm(t)), pan=pan, tilt=tilt, roll=roll,
                rot=math.degrees(float(np.linalg.norm(rvec))), parallax=parallax)


# --------------------------------------------------------------------------- html

CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--code:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--code:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--code:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,
"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 4px}h2{font-size:18px;margin:28px 0 8px;padding-top:8px;
border-top:1px solid var(--border)}h3{font-size:15px;margin:18px 0 6px}
p,li{color:var(--ink2);max-width:110ch}code{background:var(--code);padding:1px 4px;border-radius:3px}
a{color:#3987e5}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
margin:10px 0}
.tiles{display:flex;flex-wrap:wrap;gap:10px}.tile{background:var(--surface);border:1px solid
var(--border);border-radius:8px;padding:10px 14px;min-width:140px}.tile b{display:block;
font-size:24px}.tile span{color:var(--ink2);font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
vertical-align:baseline}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.grid.wide{grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
figure{margin:0;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px}figure img{width:100%;display:block;border-radius:4px;image-rendering:auto}
figcaption{font-size:12.5px;color:var(--ink2);margin-top:6px}figcaption b{color:var(--ink)}
.blink{position:relative}.blink img.b{position:absolute;inset:0;opacity:0}
.blink:hover img.b,.blinking .blink img.b.on{opacity:1}
table{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:3px 6px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);color:var(--ink2);font-weight:600}
td.l,th.l{text-align:left}tr:hover td{background:var(--code)}
.kv td{text-align:left}.kv td:first-child{color:var(--ink2);padding-right:14px}
.bad{color:#d03b3b;font-weight:600}.ok{color:#0ca30c;font-weight:600}
.nav{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;display:flex;gap:14px;
align-items:center;border-bottom:1px solid var(--border);flex-wrap:wrap}
.strip{display:flex;flex-wrap:wrap;gap:2px}.strip a{display:block;width:12px;height:22px;
border-radius:2px}.strip a.kf{box-shadow:inset 0 -4px 0 var(--ink)}
svg text{fill:var(--ink2);font-size:11px}svg .ax{stroke:var(--axis)}svg .gr{stroke:var(--grid)}
.cbar{height:10px;border-radius:2px;margin-top:4px}
.cbl{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.scroll{overflow-x:auto}
"""

BLINK_JS = """
<script>
(function(){let on=false,t=null;const cb=document.getElementById('autoblink');
if(cb){cb.addEventListener('change',()=>{document.body.classList.toggle('blinking',cb.checked);
clearInterval(t);if(cb.checked){t=setInterval(()=>{on=!on;document.querySelectorAll('.blink img.b')
.forEach(i=>i.classList.toggle('on',on));},600);}});}
document.addEventListener('keydown',e=>{const a=document.getElementById(e.key==='ArrowLeft'?'prev':
e.key==='ArrowRight'?'next':'');if(a&&a.href)location.href=a.href;});})();
</script>
"""


def esc(s):
    return html.escape(str(s))


def page(title, body, css_path='style.css'):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><link rel="stylesheet" href="{css_path}"></head>'
            f'<body><main>{body}</main>{BLINK_JS}</body></html>')


def fig(src, caption, href=None):
    href = href or src
    return (f'<figure><a href="{esc(href)}"><img loading="lazy" src="{esc(src)}"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def fig_blink(src_a, src_b, caption):
    """Image A; hovering (or auto-blink) shows B on top: a blink comparator."""
    return (f'<figure><div class="blink"><img loading="lazy" src="{esc(src_a)}">'
            f'<img loading="lazy" class="b" src="{esc(src_b)}"></div>'
            f'<figcaption>{caption}</figcaption></figure>')


def cbar(lo_label, hi_label, mid_label='0'):
    return ('<div class="cbar" style="background:linear-gradient(90deg,#2a78d6,#808080,#e34948)">'
            f'</div><div class="cbl"><span>{lo_label}</span><span>{mid_label}</span>'
            f'<span>{hi_label}</span></div>')


def kv_table(rows):
    return '<table class="kv">' + ''.join(
        f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in rows) + '</table>'


def legend():
    return ' &nbsp; '.join(f'<span><span class="sw" style="background:{c}"></span>{esc(l)}'
                           '</span>' for l, c in CATS.values())


def f2(v, fmt='{:.2f}'):
    return '' if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v)


# --------------------------------------------------------------------------- svg charts

def nice_step(raw):
    e = 10 ** math.floor(math.log10(max(raw, 1e-12)))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * e:
            return m * e
    return 10 * e


def svg_chart(recs, key, title, ylabel, hlines=(), log=False, extra=None, height=170):
    """One metric over frame index; dots coloured by status, gate lines dashed."""
    W, H, ml, mr, mt, mb = 720, height, 52, 10, 22, 26
    pts = [(r['n'], r[key], r) for r in recs if r.get(key) is not None
           and math.isfinite(r[key])]
    n_max = max([r['n'] for r in recs] + [1])
    vals = [p[1] for p in pts] + [h[0] for h in hlines]
    if extra:
        vals += [r[extra[0]] for r in recs if r.get(extra[0]) is not None]
    if not vals:
        return ''
    if log:
        vals = [max(v, 1e-1) for v in vals]
        lo, hi = math.log10(max(min(vals), 0.1)), math.log10(max(vals) * 1.5)
        tr = lambda v: math.log10(max(v, 0.1))  # noqa: E731
    else:
        lo, hi = min(0.0, min(vals)), max(vals) * 1.08 if max(vals) > 0 else 1.0
        tr = lambda v: v  # noqa: E731
    hi = hi if hi > lo else lo + 1

    def X(n):
        return ml + (W - ml - mr) * n / max(n_max, 1)

    def Y(v):
        return mt + (H - mt - mb) * (1 - (tr(v) - lo) / (hi - lo))
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(title)}">',
         f'<text x="{ml}" y="13" style="font-weight:600;fill:var(--ink)">{esc(title)} '
         f'<tspan style="font-weight:400;fill:var(--ink2)">[{esc(ylabel)}]</tspan></text>']
    if log:
        ticks = [10 ** e for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1)]
    else:
        step = nice_step((hi - lo) / 4)
        ticks = list(np.arange(math.ceil(lo / step) * step, hi + 1e-9, step))
    for tv in ticks:
        if lo - 1e-9 <= tr(tv) <= hi + 1e-9:
            y = Y(tv)
            o.append(f'<line class="gr" x1="{ml}" x2="{W - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
            lab = f'{tv:.0e}'.replace('e+0', 'e') if log and tv >= 1e4 else f'{tv:.4g}'
            o.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{lab}</text>')
    o.append(f'<line class="ax" x1="{ml}" x2="{W - mr}" y1="{H - mb}" y2="{H - mb}"/>')
    for k in range(0, n_max + 1, max(1, n_max // 10)):
        o.append(f'<text x="{X(k):.1f}" y="{H - 8}" text-anchor="middle">{k}</text>')
    for v, lab in hlines:
        y = Y(v)
        o.append(f'<line x1="{ml}" x2="{W - mr}" y1="{y:.1f}" y2="{y:.1f}" '
                 f'stroke="var(--ink2)" stroke-dasharray="4 3" stroke-width="1"/>')
        o.append(f'<text x="{W - mr - 2}" y="{y - 3:.1f}" text-anchor="end">{esc(lab)}</text>')
    if extra:
        for r in recs:
            v = r.get(extra[0])
            if v is not None and math.isfinite(v):
                o.append(f'<line x1="{X(r["n"]) - 3:.1f}" x2="{X(r["n"]) + 3:.1f}" '
                         f'y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--ink2)" stroke-width="2"/>')
    for n, v, r in pts:
        col = CATS[r['cat']][1]
        tip = f'frame {n}: {ylabel} {v:.3g} ({CATS[r["cat"]][0]})'
        o.append(f'<a href="frames/f{n:05d}.html"><g><title>{esc(tip)}</title>'
                 f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="8" fill="transparent"/>'
                 f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" '
                 f'stroke="var(--surface)" stroke-width="1.5"/></g></a>')
    o.append('</svg>')
    return ''.join(o)


def svg_map(recs, kfs, W=1400):
    """Top-down map (map x/y, equal scale): keyframes (squares), frames coloured
    by status, a line from each matched frame to its keyframe, heading ticks."""
    xy = np.array([r['pos'][:2] for r in recs] + [k['pos'][:2] for k in kfs.values()])
    lo, hi = xy.min(0), xy.max(0)
    c = (lo + hi) / 2
    m = 30
    span_x = max(float(hi[0] - lo[0]), 1e-3) * 1.05
    span_y = max(float(hi[1] - lo[1]), 1e-3) * 1.05
    scale = min((W - 2 * m) / span_x, 900 / span_y)      # px per metre
    H = int(max(span_y * scale, 220)) + 2 * m

    def P(p):
        return (W / 2 + scale * (p[0] - c[0]), H / 2 - scale * (p[1] - c[1]))
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="top-down map">']
    # scale bar
    step = nice_step((W - 2 * m) / scale / 8)
    x0, y0 = m, H - 10
    sl = scale * step
    o.append(f'<line class="ax" x1="{x0}" x2="{x0 + sl:.1f}" y1="{y0}" y2="{y0}" '
             f'stroke-width="2"/><text x="{x0 + sl + 6:.1f}" y="{y0 + 4}">{step:g} m</text>')
    for r in recs:
        if r['kf'] is not None and r['kf'] in kfs:
            a, b = P(r['pos']), P(kfs[r['kf']]['pos'])
            o.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                     f'stroke="{CATS[r["cat"]][1]}" stroke-opacity=".5" stroke-width="1"/>')
    for k in kfs.values():
        x, y = P(k['pos'])
        o.append(f'<g><title>keyframe {k["id"]} (stored at frame {k["frame"]})'
                 f'</title><rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                 f'fill="none" stroke="var(--ink2)" stroke-width="1.2"/></g>')
    for r in recs:
        x, y = P(r['pos'])
        hx, hy = r['axis'][:2] / (np.linalg.norm(r['axis'][:2]) + 1e-9)
        col = CATS[r['cat']][1]
        tip = f'frame {r["n"]}: {r["status"]}'
        o.append(f'<a href="frames/f{r["n"]:05d}.html"><g><title>{esc(tip)}</title>'
                 f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 12 * hx:.1f}" y2="{y - 12 * hy:.1f}" '
                 f'stroke="{col}" stroke-width="1.5"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="transparent"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}" '
                 f'stroke="var(--surface)" stroke-width="1"/></g></a>')
    o.append('</svg>')
    return ''.join(o)


def svg_hist(values, sigma, z_low, z_high, W=460, H=150):
    """Histogram of the plain residual with the noise sigma and z thresholds."""
    lim = max(z_high * sigma * 1.6, float(np.percentile(np.abs(values), 99.5)))
    hist, edges = np.histogram(np.clip(values, -lim, lim), bins=100, range=(-lim, lim))
    hv = np.log1p(hist)
    ml, mb, mt = 8, 22, 8

    def X(v):
        return ml + (W - 2 * ml) * (v + lim) / (2 * lim)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="residual histogram">']
    bw = (W - 2 * ml) / len(hist)
    for i, v in enumerate(hv):
        h = (H - mb - mt) * v / max(hv.max(), 1e-9)
        o.append(f'<rect x="{ml + i * bw:.1f}" y="{H - mb - h:.1f}" width="{max(bw - 1, 1):.1f}" '
                 f'height="{h:.1f}" fill="#3987e5" rx="1"/>')
    for k, lab in ((1, 'σ'), (z_low, f'{z_low:g}σ'), (z_high, f'{z_high:g}σ')):
        for s in (-1, 1):
            x = X(s * k * sigma)
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{H - mb}" '
                     f'stroke="var(--ink2)" stroke-dasharray="3 3"/>')
            o.append(f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle">'
                     f'{"-" if s < 0 else ""}{lab}</text>')
    o.append(f'<line class="ax" x1="{ml}" x2="{W - ml}" y1="{H - mb}" y2="{H - mb}"/></svg>')
    return ''.join(o)


# --------------------------------------------------------------------------- per frame

def explain_noref(cands, cfg):
    """Why no keyframe matched: the best few candidates and their failing test."""
    if not cands:
        return '<p>The keyframe map is empty (first frame).</p>'
    rows = []
    for c in cands:
        fails = []
        if c['d'] > cfg.r_match:
            fails.append(f'distance {c["d"]:.2f} m &gt; r_match {cfg.r_match:g}')
        if c['a'] > cfg.theta_match_deg:
            fails.append(f'angle {c["a"]:.1f}° &gt; theta_match {cfg.theta_match_deg:g}')
        if c['age'] < cfg.min_keyframe_age_s:
            fails.append(f'age {c["age"]:.1f} s &lt; min_keyframe_age_s '
                         f'{cfg.min_keyframe_age_s:g}')
        rows.append(f'<tr><td>kf{c["id"]}</td><td>{c["d"]:.2f}</td><td>{c["a"]:.1f}</td>'
                    f'<td>{c["age"]:.1f}</td><td class="l">{"; ".join(fails)}</td></tr>')
    return ('<p>Nearest keyframes (by distance) and the retrieval test each fails:</p>'
            '<div class="scroll"><table><tr><th>kf</th><th>d [m]</th><th>angle [°]</th>'
            '<th>age [s]</th><th class="l">fails</th></tr>' + ''.join(rows) +
            '</table></div>')


def frame_report(n, res, cfg, img_dir, rel, cands):
    """Write all images of one frame and return the HTML body sections."""
    dbg = res.debug
    cur, valid = dbg['cur'], dbg['valid']
    h, w = cur.shape[:2]
    cur8, cur_lims = norm8(cur, valid, cfg.viz.gray_low_pct, cfg.viz.gray_high_pct)
    p = f'f{n:05d}_'
    S = []

    def im(name, img, png=False):
        return rel + save(img_dir, p + name + ('.png' if png else '.jpg'), img, png)

    # ---------------------------------------------------------------- retrieval / pose
    S.append('<h2>1. Retrieval and pose change</h2>')
    if res.match is None:
        S.append(explain_noref(cands, cfg.store))
        S.append('<div class="grid">' + fig(im('cur', cur8), '<b>current</b> (undistorted)')
                 + fig(im('overlay', res.overlay), '<b>final overlay</b> '
                       '<code>~/overlay</code>') + '</div>')
        if res.inserted is not None:
            S.append(f'<p>Stored as new keyframe <b>kf{res.inserted.id}</b> (no keyframe within '
                     f'r_insert={cfg.store.r_insert:g} m and theta_insert='
                     f'{cfg.store.theta_insert_deg:g}°).</p>')
        S.append('<h2>2. Final overlay</h2><div class="grid">'
                 + fig(im('overlay', res.overlay), 'node output <code>~/overlay</code>')
                 + '</div>')
        return S

    kf = res.match.keyframe
    cur_src = im('cur', cur8)
    overlay_src = im('overlay', res.overlay)
    ref = dbg['ref']
    ref8, _ = norm8(ref, valid, cfg.viz.gray_low_pct, cfg.viz.gray_high_pct)
    pc = pose_change(kf.T_map_cam, dbg['T_map_cam'], dbg['K'])
    t = pc['t']
    par = ', '.join(f'{v:.0f} px @ {Z:g} m' for Z, v in pc['parallax'].items())
    S.append('<div class="grid">'
             + fig(im('kf', ref8), f'<b>keyframe kf{kf.id}</b> (recorded '
                   f'{res.stamp - kf.stamp:.1f} s earlier)')
             + fig(cur_src, f'<b>current frame {n}</b>; both undistorted, each contrast-stretched')
             + fig(overlay_src, '<b>final overlay</b> <code>~/overlay</code> (details in 4.)')
             + '<div class="card" style="grid-column:1/-1">' + kv_table([
                 ('retrieval distance / axis angle',
                  f'{res.match.distance:.3f} m / {res.match.angle_deg:.2f}°  '
                  f'(limits {cfg.store.r_match:g} m / {cfg.store.theta_match_deg:g}°)'),
                 ('translation in kf camera [x right, y down, z fwd]',
                  f'{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f} m'),
                 ('rotation total', f'{pc["rot"]:.2f}°'),
                 ('pan / tilt / roll', f'{pc["pan"]:+.2f}° / {pc["tilt"]:+.2f}° / '
                                       f'{pc["roll"]:+.2f}°'),
                 ('rotation-only prior error (est.)', par),
             ]) + '<p style="font-size:12px">The pose prior is a pure-rotation homography '
             '<code>K R K⁻¹</code>; the translation is ignored, so a scene at depth Z is '
             'misplaced by roughly the parallax above. It must be corrected by ECC, and that '
             f'correction is gated at max_correction_px={cfg.align.max_correction_px:g}.</p>'
             '</div></div>')

    # ---------------------------------------------------------------- alignment
    ar = res.alignment
    W0 = ar.W0
    cur_p = preprocess(cur, valid, cfg.align.use_gradient)
    ref_p = preprocess(ref, valid, cfg.align.use_gradient)
    ref_m = (valid > 0).astype(np.uint8)
    stages = [('pose prior (rotation homography)', W0, '')]
    for tr in dbg.get('align_trace') or []:
        err = tr['error'].split('error: ')[-1].split(' in function')[0]
        stages.append((f'after ECC level {tr["level"]} (1/{int(round(1 / tr["scale"]))} res)',
                       tr['W'], err))

    S.append('<h2>2. Alignment: pose prior → ECC</h2>')
    S.append('<p>Each row warps the keyframe into the current frame. <b>Blend</b>: current = '
             'green, keyframe = magenta, gray where they agree; double edges = misalignment. '
             '<b>Hover</b> the warped keyframe to flip to the current image (or enable '
             'auto-blink at the top). <b>|diff|</b>: absolute difference of the normalised '
             'images ECC works on (white = 3 std). ρ is the masked correlation the gate '
             'uses, computed at full resolution.</p>')
    for i, (name, W, err) in enumerate(stages):
        ref_w8 = warp_to_current(ref8, W, (w, h))
        m_w = (warp_to_current(ref_m, W, (w, h), cv2.INTER_NEAREST) > 0) & (valid > 0)
        ref_pw = warp_to_current(ref_p, W, (w, h))
        rho = masked_correlation(cur_p, ref_pw, m_w)
        ov = float(m_w.sum()) / max(1, int((valid > 0).sum()))
        shift = _centre_shift(W, W0, w, h)
        diff = np.clip(np.abs(cur_p - ref_pw) * (255 / 3.0), 0, 255).astype(np.uint8)
        a = im(f'st{i}_warp', hatch_invalid(gray_bgr(ref_w8), m_w))
        cap = (f'ρ={rho:.3f} · overlap={ov:.2f} · correction={shift:.0f} px'
               + (f' · <span class="bad">ECC error: {esc(err)}</span>' if err else ''))
        S.append(f'<h3>{esc(name)}</h3><div class="grid">'
                 + fig_blink(a, cur_src, f'<b>warped keyframe</b> (hover → current) · {cap}')
                 + fig(im(f'st{i}_blend', blend_mg(cur8, ref_w8, m_w)), '<b>blend</b> '
                       'current=green / keyframe=magenta')
                 + fig(im(f'st{i}_diff', hatch_invalid(gray_bgr(diff), m_w)),
                       '<b>|diff|</b> of normalised images')
                 + '</div>')

    pad = 200
    fp_kf = footprint_canvas(ref8, [(W0, C_PRIOR, 'prior'), (ar.W, C_ECC, 'ECC')], w, h, pad)
    corner_arrows(fp_kf, W0, ar.W, w, h, pad)
    fp_cur = footprint_canvas(cur8, [(np.linalg.inv(W0), C_PRIOR, 'prior'),
                                     (np.linalg.inv(ar.W), C_ECC, 'ECC')], w, h, pad)
    gate = [
        ('overlap', f'{ar.overlap:.3f}', f'≥ {cfg.align.min_overlap:g}',
         ar.overlap >= cfg.align.min_overlap),
        ('correlation ρ', f'{ar.correlation:.3f}', f'≥ {cfg.align.min_correlation:g}',
         ar.correlation >= cfg.align.min_correlation),
        ('correction', f'{ar.correction_px:.1f} px', f'≤ {cfg.align.max_correction_px:g} px',
         ar.correction_px <= cfg.align.max_correction_px),
    ]
    gate_rows = ''.join(
        f'<tr><td class="l">{k}</td><td>{v}</td><td>{lim}</td>'
        f'<td class="l {"ok" if ok else "bad"}">{"pass" if ok else "FAIL"}</td></tr>'
        for k, v, lim, ok in gate)
    verdict = ('<span class="ok">ACCEPTED</span>' if ar.ok
               else f'<span class="bad">REJECTED: {esc(ar.reason)}</span>')
    S.append('<h3>Footprints and quality gate</h3><div class="grid wide">'
             + fig(im('fp_kf', fp_kf), '<b>current image footprint in the keyframe</b>: '
                   '<span style="color:#ffbe00">yellow = prior</span>, '
                   '<span style="color:#28c8ff">cyan = ECC</span>; arrows = correction '
                   'at the corners and the centre (the gate uses the centre)')
             + fig(im('fp_cur', fp_cur), '<b>keyframe footprint in the current image</b> '
                   '(same colours)')
             + '<div class="card"><table><tr><th class="l">check</th><th>value</th>'
               '<th>limit</th><th class="l">result</th></tr>' + gate_rows + '</table>'
             f'<p>Verdict: {verdict}. Checks run in this order; the first failure is '
             'reported. When rejected, no change map is computed.</p></div></div>')

    if res.change is None:
        S.append('<h2>3. Change detection</h2><p>Skipped: alignment rejected.</p>')
        S.append('<h2>4. Final overlay</h2><div class="grid">'
                 + fig(overlay_src, 'node output <code>~/overlay</code>')
                 + '</div>')
        return S

    # ---------------------------------------------------------------- change
    ch, ct, cc = res.change, dbg['change_trace'], cfg.change
    v = ct['valid']
    sig = ch.sigma
    ref_w = dbg['ref_warped']
    refw8, _ = norm8(ref_w, v, lims=cur_lims)
    refa8, _ = norm8(ch.ref_adjusted, v, lims=cur_lims)
    vmax = cc.z_high * sig
    S.append('<h2>3. Change detection</h2>')

    idx = np.flatnonzero(v)
    if idx.size > 60000:
        idx = np.random.default_rng(0).choice(idx, 60000, replace=False)
    dens, (dlo, dhi) = gain_density(ref_w.ravel()[idx], cur.ravel()[idx], ch.gain, ch.offset)
    S.append('<h3>3a. Overlap mask and gain/offset fit</h3><div class="grid">'
             + fig_blink(im('c_refw', hatch_invalid(gray_bgr(refw8), v)), cur_src,
                         '<b>warped keyframe before gain fit</b>, same grey scale as current '
                         '(hover → current). Hatched = outside the eroded overlap')
             + fig_blink(im('c_refa', hatch_invalid(gray_bgr(refa8), v)), cur_src,
                         f'<b>after gain/offset</b> cur ≈ {ch.gain:.4f}·ref {ch.offset:+.1f} '
                         '(hover → current)')
             + fig(im('c_gain', dens, png=True),
                   f'<b>robust fit</b>: x = keyframe, y = current, range {dlo:.0f}…{dhi:.0f}. '
                   'Red = fit, gray = identity. Off-line clusters are changes or misaligned '
                   'pixels.')
             + '</div>')

    plain = np.where(v, ct['plain'], np.nan)
    S.append('<h3>3b. Residuals and noise level</h3><div class="grid">'
             + fig(im('c_plain', diverging(plain, vmax, v)),
                   '<b>plain residual</b> blur(cur − adjusted ref): its robust spread is the '
                   f'noise σ = {sig:.2f} (floor min_sigma={cc.min_sigma:g})'
                   + cbar(f'−{vmax:.0f}', f'+{vmax:.0f}  (= ±z_high·σ)'))
             + fig(im('c_nbr', diverging(np.where(v, ct['nbhd_raw'], np.nan), vmax, v)),
                   f'<b>neighbourhood difference</b> (radius {cc.nbhd_radius} px): zero where '
                   'the current value lies within the min…max of the reference nearby, so '
                   'edges shifted by ≤ radius vanish' + cbar(f'−{vmax:.0f}', f'+{vmax:.0f}'))
             + fig(im('c_nbb', diverging(np.where(v, ct['nbhd_blur'], np.nan), vmax, v)),
                   f'<b>smoothed difference</b> (σ_blur={cc.blur_sigma:g} px) = published '
                   '<code>~/diff</code>' + cbar(f'−{vmax:.0f}', f'+{vmax:.0f}'))
             + '<figure>' + svg_hist(ct['plain'][v], sig, cc.z_low, cc.z_high)
             + '<figcaption><b>histogram of the plain residual</b> (log counts) with ±σ and '
               'the z thresholds. Heavy tails beyond z_high are what gets detected.'
               '</figcaption></figure>'
             + '</div>')

    red, blue = (72, 73, 227), (214, 120, 42)
    lred, lblue = (150, 150, 235), (235, 180, 120)
    zc = z_classes(ch.z, v, cc.z_low, cc.z_high, cur8)
    hyst = mask_image(cur8, v, [(ct['warmer_weak'], lred), (ct['colder_weak'], lblue),
                                (ct['warmer_hyst'], red), (ct['colder_hyst'], blue)])
    opened = mask_image(cur8, v, [(ct['warmer_hyst'], lred), (ct['colder_hyst'], lblue),
                                  (ct['warmer_open'], red), (ct['colder_open'], blue)])
    blob_img = mask_image(cur8, v, [(ct['warmer_open'], lred), (ct['colder_open'], lblue)])
    for b in ch.blobs:
        x, y, bw, bh = b.bbox
        col = red if b.kind == 'warmer' else blue
        cv2.rectangle(blob_img, (x, y), (x + bw - 1, y + bh - 1), col, 2)
        cv2.putText(blob_img, f'{b.area}px z{b.peak_z:.0f}', (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    S.append('<h3>3c. Thresholding → blobs</h3><div class="grid">'
             + fig(im('c_z', zc), f'<b>z = diff / σ</b>: light = z_low ≤ |z| &lt; z_high '
                   f'({cc.z_low:g}…{cc.z_high:g}), solid = |z| ≥ z_high. '
                   'Red = warmer, blue = colder')
             + fig(im('c_hyst', hyst), '<b>hysteresis</b>: weak regions (light) are kept '
                   '(solid) only if they contain a strong pixel')
             + fig(im('c_open', opened), f'<b>opening</b> (kernel {cc.open_kernel}): light '
                   'parts are removed as too thin')
             + fig(im('c_blobs', blob_img), f'<b>connected components</b> ≥ min_blob_area '
                   f'({cc.min_blob_area} px) become blobs; smaller ones are dropped')
             + '</div>')

    rows = ''.join(
        f'<tr><td class="l">{b.kind}</td><td>{b.area}</td><td>{b.bbox}</td>'
        f'<td>{b.mean_value:+.1f}</td><td>{b.peak_value:+.1f}</td><td>{b.peak_z:.1f}</td>'
        f'<td>{b.persistence}</td><td class="l {"ok" if b.confirmed else ""}">'
        f'{"confirmed" if b.confirmed else "pending"}</td></tr>' for b in res.blobs)
    S.append('<h2>4. Persistence and final overlay</h2><div class="grid wide">'
             + fig(overlay_src, 'node output <code>~/overlay</code> '
                   '(thick box = confirmed detection, thin = not yet persistent)')
             + '<div class="card"><p>A blob is confirmed after '
               f'{cfg.persistence.required_frames} consecutive frames (matched via the rotation '
               f'homography, ±{cfg.persistence.radius_px} px). Only confirmed blobs are '
               'published on <code>~/detections</code>.</p>'
             + ('<div class="scroll"><table><tr><th class="l">kind</th><th>area</th><th>bbox</th>'
                '<th>mean Δ</th><th>peak Δ</th><th>peak z</th><th>frames</th>'
                f'<th class="l">state</th></tr>{rows}</table></div>' if rows
                else '<p>No blobs in this frame.</p>')
             + '</div></div>')
    return S


# --------------------------------------------------------------------------- main

def categorize(res):
    if res.match is None:
        return 'noref'
    a = res.alignment
    if a.ok:
        return 'accepted'
    if a.reason.startswith('overlap'):
        return 'rej_overlap'
    if a.reason.startswith('correlation'):
        return 'rej_corr'
    return 'rej_shift'


def parse_range(s):
    if not s:
        return None
    a, _, b = s.partition(':')
    return (int(a) if a else 0, int(b) if b else 10 ** 9)


def nearest_candidates(store, T, stamp, k=3):
    if len(store) == 0:
        return []
    d, a = store._dist_angle(T)
    order = np.argsort(d)[:k]
    return [dict(id=store._kfs[i].id, d=float(d[i]), a=float(a[i]),
                 age=float(stamp - store._stamps[i])) for i in order]


def write_index(out, args, cfg, recs, kfs):
    counts = {c: sum(r['cat'] == c for r in recs) for c in CATS}
    n_match = len(recs) - counts['noref']
    tiles = (f'<div class="tile"><b>{len(recs)}</b><span>processed frames</span></div>'
             f'<div class="tile"><b>{len(kfs)}</b><span>keyframes stored</span></div>'
             + ''.join(f'<div class="tile"><b>{counts[c]}</b><span><span class="sw" '
                       f'style="background:{col}"></span>{esc(lab)}</span></div>'
                       for c, (lab, col) in CATS.items()))
    strip = ''.join(
        f'<a href="frames/f{r["n"]:05d}.html" class="{"kf" if r["inserted"] is not None else ""}"'
        f' style="background:{CATS[r["cat"]][1]}" title="frame {r["n"]}: {esc(r["status"])}">'
        '</a>' for r in recs)
    a = cfg.align
    charts = (
        svg_chart(recs, 'd', 'Distance to matched keyframe', 'm',
                  [(cfg.store.r_match, f'r_match {cfg.store.r_match:g}')])
        + svg_chart(recs, 'angle', 'Optical-axis angle to matched keyframe', 'deg',
                    [(cfg.store.theta_match_deg, f'theta_match {cfg.store.theta_match_deg:g}')])
        + svg_chart(recs, 'rho', 'Correlation ρ after ECC (dot) and with the pose prior '
                    'only (dash)', 'ρ', [(a.min_correlation, f'min {a.min_correlation:g}')],
                    extra=('rho_prior',))
        + svg_chart(recs, 'shift', 'ECC correction vs. pose prior (centre shift, log)',
                    'px', [(a.max_correction_px, f'max {a.max_correction_px:g}')], log=True)
        + svg_chart(recs, 'sigma', 'Noise σ (accepted frames)', 'counts')
        + svg_chart(recs, 'n_blobs', 'Blobs per frame (accepted frames)', 'blobs'))
    rows = []
    for r in recs:
        col = CATS[r['cat']][1]
        rows.append(
            f'<tr><td><a href="frames/f{r["n"]:05d}.html">{r["n"]}</a></td>'
            f'<td>{r["t"] - recs[0]["t"]:.1f}</td><td class="l"><span class="sw" style="background:{col}">'
            f'</span>{esc(CATS[r["cat"]][0])}</td>'
            f'<td>{"" if r["kf"] is None else "kf%d" % r["kf"]}</td><td>{f2(r["d"])}</td>'
            f'<td>{f2(r["angle"], "{:.1f}")}</td><td>{f2(r["rho_prior"])}</td>'
            f'<td>{f2(r["rho"])}</td><td>{f2(r["shift"], "{:.0f}")}</td>'
            f'<td>{f2(r["overlap"])}</td><td>{f2(r["ecc_fail"], "{:d}")}</td><td>{f2(r["gain"], "{:.3f}")}</td>'
            f'<td>{f2(r["sigma"], "{:.1f}")}</td><td>{f2(r["n_blobs"], "{:d}")}</td>'
            f'<td>{f2(r["n_conf"], "{:d}")}</td>'
            f'<td>{"" if r["inserted"] is None else "kf%d" % r["inserted"]}</td></tr>')
    body = f"""
<h1>Thermal change detection: debug report</h1>
<p><code>{esc(args.bag)}</code> · image <code>{esc(args.image)}</code> · map frame
<code>{esc(args.map_frame)}</code> · params <code>{esc(args.params or 'defaults')}</code></p>
<div class="tiles">{tiles}</div>
<div class="card"><b>How to read the statuses</b><ul>
<li><b>no reference</b>: no stored keyframe is within r_match={cfg.store.r_match:g} m
 <i>and</i> theta_match={cfg.store.theta_match_deg:g}° <i>and</i> at least
 min_keyframe_age_s={cfg.store.min_keyframe_age_s:g} s old. Normal on the first lap.</li>
<li><b>REJECTED</b>: a keyframe was found, the keyframe was aligned to the current image
 (rotation homography from the poses, refined by ECC), but the alignment failed the
 quality gate, so no change map is computed for that frame:
 <ul><li><b>overlap</b> &lt; {a.min_overlap:g}: too little of the image is covered by the warped keyframe;</li>
 <li><b>correlation</b> ρ &lt; {a.min_correlation:g}: the aligned images do not look alike
 (wrong keyframe, ECC diverged, or the scene really changed a lot);</li>
 <li><b>correction</b> &gt; {a.max_correction_px:g} px: ECC moved the image centre further than
 this from where the pose prior put it. Either the prior is poor (translation /
 parallax is ignored by the rotation-only homography, TF timing) or ECC
 converged to a wrong / degenerate homography.</li></ul></li>
<li><b>accepted</b>: the change map is computed and blobs are reported.</li></ul>
Every frame page shows the pose change, each ECC pyramid level and the gate values.</div>
<h2>Timeline</h2><p>{legend()} &nbsp; (dark underline = frame stored as keyframe). Click a
cell to open the frame.</p><div class="strip">{strip}</div>
<h2>Top-down map</h2><p>Frames (dots, heading tick = optical axis) coloured by status,
keyframes as squares, thin line = matched keyframe. {n_match} of {len(recs)} frames
found a reference.</p><div class="card">{svg_map(recs, kfs)}</div>
<h2>Metrics per frame</h2><p>{legend()}</p><div class="card">{charts}</div>
<h2>All frames</h2><div class="card scroll"><table><tr><th>frame</th><th>t [s]</th>
<th class="l">status</th><th>kf</th><th>d [m]</th><th>angle [°]</th><th>ρ prior</th>
<th>ρ ECC</th><th>corr. [px]</th><th>overlap</th><th>ECC failed levels</th><th>gain</th><th>σ</th><th>blobs</th>
<th>confirmed</th><th>new kf</th></tr>{''.join(rows)}</table></div>
"""
    with open(os.path.join(out, 'index.html'), 'w') as f:
        f.write(page('Thermal change debug', body))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bag_args(ap)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--detail', default='',
                    help='frame index range START:END for per-frame pages (default: all)')
    args = ap.parse_args()

    cfg = load_ros_params_yaml(args.params) if args.params else PipelineConfig()
    cfg.store.cleanup_on_exit = True
    detail = parse_range(args.detail)
    out = args.out
    img_dir = os.path.join(out, 'img')
    frames_dir = os.path.join(out, 'frames')
    for d in (img_dir, frames_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    with open(os.path.join(out, 'style.css'), 'w') as f:
        f.write(CSS)

    print('pass 1: loading TF ...')
    tf_buffer = load_tf(args.bag)
    print('pass 2: processing images ...')
    pipeline = ChangeDetectionPipeline(cfg)
    recs, kfs, pages = [], {}, {}
    try:
        for n, (t, img, T, info) in enumerate(iter_frames(args, tf_buffer)):
            K = np.array(info.k).reshape(3, 3)
            cands = nearest_candidates(pipeline.store, T, t)
            res = pipeline.process(img, t, T, K, np.array(info.d), info.distortion_model,
                                   debug=True)
            cat = categorize(res)
            m, a, c = res.match, res.alignment, res.change
            rho_prior = None
            if a is not None:
                dbg = res.debug
                h, w = dbg['cur'].shape[:2]
                cp = preprocess(dbg['cur'], dbg['valid'], cfg.align.use_gradient)
                rp = warp_to_current(preprocess(dbg['ref'], dbg['valid'], cfg.align.use_gradient),
                                     a.W0, (w, h))
                mw = (warp_to_current(dbg['valid'], a.W0, (w, h), cv2.INTER_NEAREST) > 0) \
                    & (dbg['valid'] > 0)
                rho_prior = masked_correlation(cp, rp, mw)
            rec = dict(
                n=n, t=t, status=res.status, cat=cat, pos=T[:3, 3].copy(),
                axis=optical_axis(T).copy(), kf=m.keyframe.id if m else None,
                d=m.distance if m else None, angle=m.angle_deg if m else None,
                rho=a.correlation if a else None, rho_prior=rho_prior,
                shift=a.correction_px if a else None, overlap=a.overlap if a else None,
                gain=c.gain if c else None, sigma=c.sigma if c else None,
                n_blobs=len(c.blobs) if c else None,
                n_conf=sum(b.confirmed for b in c.blobs) if c else None,
                inserted=res.inserted.id if res.inserted else None,
                ecc_fail=(sum(bool(tr['error']) for tr in res.debug.get('align_trace') or [])
                          if a else None))
            recs.append(rec)
            if res.inserted is not None:
                kfs[res.inserted.id] = dict(id=res.inserted.id, pos=T[:3, 3].copy(), t=t,
                                            frame=n)
            if detail is None or detail[0] <= n < detail[1]:
                pages[n] = frame_report(n, res, cfg, img_dir, '../img/', cands)
            print(f'  [{n}] {res.status}')
            if args.max_frames and n + 1 >= args.max_frames:
                break
    finally:
        pipeline.close()

    order = sorted(pages)
    for i, n in enumerate(order):
        r = recs[n]
        prv = f'f{order[i - 1]:05d}.html' if i > 0 else ''
        nxt = f'f{order[i + 1]:05d}.html' if i + 1 < len(order) else ''
        col = CATS[r['cat']][1]
        nav = (f'<div class="nav"><a href="../index.html">← index</a>'
               f'<a id="prev" href="{prv}">‹ prev</a><a id="next" href="{nxt}">next ›</a>'
               '<label><input type="checkbox" id="autoblink"> auto-blink</label>'
               '<span style="color:var(--muted)">← / → keys</span></div>')
        head = (f'<h1>Frame {n} · t={r["t"] - recs[0]["t"]:.2f} s</h1><p><span class="sw" '
                f'style="background:{col}"></span><b>{esc(CATS[r["cat"]][0])}</b> · '
                f'<code>{esc(r["status"])}</code></p>')
        with open(os.path.join(frames_dir, f'f{n:05d}.html'), 'w') as f:
            f.write(page(f'Frame {n}', nav + head + ''.join(pages[n]), '../style.css'))
    write_index(out, args, cfg, recs, kfs)
    print(f'done: {len(recs)} frames, {len(pages)} detail pages -> '
          f'{os.path.join(os.path.abspath(out), "index.html")}')


if __name__ == '__main__':
    main()
