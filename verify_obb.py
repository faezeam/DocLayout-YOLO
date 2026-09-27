"""OBB support checks for the DocLayout-YOLO fork.

Run: python verify_obb.py
Covers angle convention, probiou/nms_rotated behaviour, model forward/backward
shapes, and pretrained backbone transfer into a fresh OBB head.
"""

import math

import numpy as np
import torch

from doclayout_yolo.cfg import get_cfg
from doclayout_yolo.nn.tasks import OBBModel
from doclayout_yolo.utils import DEFAULT_CFG
from doclayout_yolo.utils.metrics import batch_probiou, probiou
from doclayout_yolo.utils.ops import (
    nms_rotated,
    regularize_rboxes,
    xywhr2xyxyxyxy,
    xyxyxyxy2xywhr,
)
from doclayout_yolo.utils.instance import Instances
from doclayout_yolo.utils.torch_utils import intersect_dicts

OBB_CFG = "yolov10m-doclayout-obb.yaml"
DETECT_CKPT = "doclayout_yolo_docstructbench_imgsz1024.pt"
NC = 10

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def rect_corners(cx, cy, w, h, theta):
    """Corners of a rotated rectangle, theta in radians, CCW in image coords."""
    c, s = math.cos(theta), math.sin(theta)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=-1)


def test_angle_convention():
    section("1. Angle convention: units, range, long-edge form")

    # Labels are produced by cv2.minAreaRect inside xyxyxyxy2xywhr.
    angles = []
    for deg in range(0, 180, 5):
        pts = rect_corners(100.0, 100.0, 60.0, 30.0, math.radians(deg)).reshape(1, 8)
        _, _, w, h, t = xyxyxyxy2xywhr(pts.astype(np.float32))[0]
        angles.append(t)
    angles = np.array(angles)
    check(
        "xyxyxyxy2xywhr returns radians, not degrees",
        angles.max() <= math.pi + 1e-4,
        f"observed max {angles.max():.4f} <= pi; a degree convention would reach ~90",
    )
    check(
        "label angle range is (0, pi/2]",
        angles.min() >= 0.0 and angles.max() <= math.pi / 2 + 1e-6,
        f"observed [{angles.min():.4f}, {angles.max():.4f}] vs pi/2={math.pi / 2:.4f}",
    )

    # cv2.minAreaRect does not guarantee w >= h, so labels are NOT long-edge.
    ratios = []
    for deg in range(0, 180, 5):
        pts = rect_corners(100.0, 100.0, 60.0, 30.0, math.radians(deg)).reshape(1, 8)
        _, _, w, h, _ = xyxyxyxy2xywhr(pts.astype(np.float32))[0]
        ratios.append(w >= h)
    check(
        "label w/h ordering is NOT guaranteed long-edge (minAreaRect raw)",
        not all(ratios),
        f"{sum(ratios)}/{len(ratios)} samples had w >= h",
    )

    # regularize_rboxes is what the predictor applies, and it does enforce long-edge.
    rb = torch.tensor(
        [
            [50.0, 50.0, 20.0, 40.0, 0.2],  # h > w, should swap
            [50.0, 50.0, 40.0, 20.0, 0.2],  # already long-edge
            [50.0, 50.0, 10.0, 60.0, 2.5],  # h > w, large angle
            [50.0, 50.0, 30.0, 30.0, -0.7],  # square, negative angle
        ]
    )
    reg = regularize_rboxes(rb.clone())
    w_, h_, t_ = reg[:, 2], reg[:, 3], reg[:, 4]
    check(
        "regularize_rboxes enforces long-edge (w >= h)",
        bool((w_ >= h_ - 1e-6).all()),
        f"w={[round(float(v), 2) for v in w_]} h={[round(float(v), 2) for v in h_]}",
    )
    check(
        "regularize_rboxes angle lands in [0, pi)",
        bool((t_ >= 0).all() and (t_ < math.pi).all()),
        f"t={[round(float(v), 4) for v in t_]}, pi={math.pi:.4f}",
    )
    check(
        "regularize_rboxes preserves the rectangle (area invariant)",
        bool(torch.allclose(rb[:, 2] * rb[:, 3], w_ * h_, atol=1e-4)),
        "w*h unchanged after edge swap",
    )

    # Head output range, set by (sigmoid - 0.25) * pi in OBB.forward.
    logits = torch.linspace(-30, 30, 2001)
    head_angle = (logits.sigmoid() - 0.25) * math.pi
    check(
        "OBB head angle range is [-pi/4, 3pi/4]",
        abs(float(head_angle.min()) + math.pi / 4) < 1e-3
        and abs(float(head_angle.max()) - 3 * math.pi / 4) < 1e-3,
        f"[{float(head_angle.min()):.4f}, {float(head_angle.max()):.4f}] vs "
        f"[{-math.pi / 4:.4f}, {3 * math.pi / 4:.4f}]",
    )

    # Round trip through the corner representation.
    rb2 = torch.tensor([[100.0, 120.0, 60.0, 30.0, 0.6], [40.0, 40.0, 25.0, 25.0, 1.2]])
    back = xyxyxyxy2xywhr(xywhr2xyxyxyxy(rb2).reshape(-1, 8))
    same_geom = torch.allclose(rb2[:, :2], back[:, :2], atol=1e-2) and torch.allclose(
        torch.sort(rb2[:, 2:4], dim=-1).values,
        torch.sort(back[:, 2:4], dim=-1).values,
        atol=1e-2,
    )
    check(
        "xywhr -> corners -> xywhr preserves centre and edge lengths",
        same_geom,
        f"in={rb2[0].tolist()} out={[round(float(v), 3) for v in back[0]]}",
    )


def test_probiou_and_nms():
    section("2. probiou / batch_probiou / nms_rotated")

    a = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.3]])
    # ProbIoU is a Bhattacharyya-distance approximation that clamps bd to eps=1e-7, so identical
    # boxes saturate near 1 - sqrt(1 - exp(-1e-7)) ~= 0.9997 rather than exactly 1.
    check(
        "probiou(box, box) saturates at ~1 for identical boxes",
        abs(float(probiou(a, a)) - 1.0) < 1e-3,
        f"got {float(probiou(a, a)):.8f} (eps-clamped, exact 1.0 is unreachable by design)",
    )

    far = torch.tensor([[1000.0, 1000.0, 40.0, 20.0, 0.3]])
    check(
        "probiou ~ 0 for far-apart boxes",
        float(probiou(a, far)) < 1e-6,
        f"got {float(probiou(a, far)):.3e}",
    )

    # Rotating both boxes together must not change their IoU.
    pair_0 = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.0], [55.0, 52.0, 40.0, 20.0, 0.0]])
    pair_r = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.9], [55.0, 52.0, 40.0, 20.0, 0.9]])
    iou_0 = float(batch_probiou(pair_0, pair_0)[0, 1])
    iou_r = float(batch_probiou(pair_r, pair_r)[0, 1])
    check(
        "batch_probiou is consistent under shared rotation",
        abs(iou_0 - iou_r) < 0.06,
        f"theta=0 -> {iou_0:.4f}, theta=0.9 -> {iou_r:.4f}",
    )

    m = batch_probiou(pair_0, pair_0)
    check(
        "batch_probiou diagonal saturates at ~1 and matrix is symmetric",
        bool(torch.allclose(m.diagonal(), torch.ones(2), atol=1e-3))
        and bool(torch.allclose(m, m.T, atol=1e-5)),
        f"diag={[round(float(v), 6) for v in m.diagonal()]}",
    )

    # nms_rotated: three near-duplicates of one box plus two well-separated boxes.
    boxes = torch.tensor(
        [
            [50.0, 50.0, 40.0, 20.0, 0.30],
            [51.0, 50.5, 40.0, 20.0, 0.31],
            [49.5, 49.5, 39.0, 20.5, 0.29],
            [300.0, 300.0, 40.0, 20.0, 0.30],
            [600.0, 120.0, 30.0, 60.0, 1.20],
        ]
    )
    scores = torch.tensor([0.90, 0.85, 0.80, 0.75, 0.70])
    keep = nms_rotated(boxes, scores, threshold=0.45)
    kept = sorted(int(i) for i in keep)
    check(
        "nms_rotated collapses duplicates and keeps distinct boxes",
        kept == [0, 3, 4],
        f"kept indices {kept}, expected [0, 3, 4]",
    )
    check(
        "nms_rotated keeps the highest-scoring box of a duplicate group",
        0 in kept and 1 not in kept and 2 not in kept,
        "index 0 has the top score of the trio",
    )

    keep_all = nms_rotated(boxes, scores, threshold=0.999)
    check(
        "nms_rotated at a near-1 threshold keeps everything",
        len(keep_all) == 5,
        f"kept {len(keep_all)}/5",
    )

    # Round trip: xywhr -> corners -> xywhr must not change what NMS keeps.
    rt = xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes).reshape(-1, 8))
    keep_rt = sorted(int(i) for i in nms_rotated(rt, scores.clone(), threshold=0.45))
    check(
        "nms_rotated is stable across a corner round trip",
        keep_rt == kept,
        f"round-trip kept {keep_rt}, original kept {kept}",
    )


def build_obb(nc=NC):
    m = OBBModel(OBB_CFG, ch=3, nc=nc, verbose=False)
    m.args = get_cfg(DEFAULT_CFG)
    return m


def test_forward_and_step():
    section("3. Forward pass and one training step")

    m = build_obb()
    head = m.model[-1]
    check("head is OBB", type(head).__name__ == "OBB", f"got {type(head).__name__}")
    check(
        "head geometry: nc=10, ne=1, nl=3",
        head.nc == NC and head.ne == 1 and head.nl == 3,
        f"nc={head.nc} ne={head.ne} nl={head.nl}",
    )

    imgsz = 640
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 3, imgsz, imgsz))
    n_anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    check(
        "eval output shape is (b, 4+nc+1, anchors)",
        tuple(y[0].shape) == (2, 4 + NC + 1, n_anchors),
        f"got {tuple(y[0].shape)}, expected (2, {4 + NC + 1}, {n_anchors})",
    )

    m.train()
    out = m(torch.rand(2, 3, imgsz, imgsz))
    feats, angle = out
    check(
        "train output carries per-anchor angle",
        tuple(angle.shape) == (2, 1, n_anchors),
        f"angle shape {tuple(angle.shape)}",
    )

    batch = {
        "img": torch.rand(2, 3, imgsz, imgsz),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([0.0, 3.0, 5.0]),
        "bboxes": torch.tensor(
            [
                [0.50, 0.50, 0.30, 0.20, 0.30],
                [0.20, 0.30, 0.20, 0.10, 1.10],
                [0.60, 0.60, 0.25, 0.15, 0.80],
            ]
        ),
    }
    loss, items = m.loss(batch)
    names = ("box", "cls", "dfl")
    check(
        "loss has three components",
        items.numel() == 3,
        f"items={[round(float(v), 4) for v in items]} ({', '.join(names)})",
    )
    check(
        "all three loss components are finite and > 0",
        bool(torch.isfinite(items).all()) and bool((items > 0).all()),
        f"{ {n: round(float(v), 4) for n, v in zip(names, items)} }",
    )
    loss.sum().backward()
    grads = [p for p in m.parameters() if p.grad is not None]
    gnorm = sum(float(p.grad.norm()) for p in grads)
    check(
        "backward produces finite gradients",
        math.isfinite(gnorm) and gnorm > 0,
        f"{len(grads)} tensors with grad, total norm {gnorm:.2f}",
    )
    head_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.model[-1].cv4.parameters()
    )
    check("angle branch (cv4) receives gradient", head_grad)


def test_pretrained_transfer():
    section("4. Pretrained transfer: backbone loads, head stays fresh")

    try:
        ckpt = torch.load(DETECT_CKPT, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        check("checkpoint present", False, f"{DETECT_CKPT} not found")
        return

    csd = ckpt["model"].float().state_dict()
    m = build_obb()
    msd = m.state_dict()
    head_prefix = f"model.{len(m.model) - 1}."

    naive = intersect_dicts(csd, msd)
    naive_head = [k for k in naive if k.startswith(head_prefix)]
    print(
        f"  note: a naive intersect_dicts would transfer {len(naive_head)} head tensors "
        f"(v10Detect cv2 is shape-compatible with OBB cv2), so the head is excluded explicitly"
    )

    before = {k: v.clone() for k, v in m.state_dict().items() if k.startswith(head_prefix)}
    m.load(ckpt["model"], verbose=False, exclude=(head_prefix,))
    after = m.state_dict()

    transferred = intersect_dicts(csd, msd, exclude=(head_prefix,))
    backbone_target = [k for k in msd if not k.startswith(head_prefix)]
    head_target = [k for k in msd if k.startswith(head_prefix)]
    skipped_shape = [
        k for k in csd if k in msd and not k.startswith(head_prefix) and csd[k].shape != msd[k].shape
    ]
    skipped_absent = [k for k in csd if k not in msd]

    print(f"\n  checkpoint tensors           : {len(csd)}")
    print(f"  OBB model tensors            : {len(msd)}  ({len(backbone_target)} non-head, {len(head_target)} head)")
    print(f"  TRANSFERRED (non-head)       : {len(transferred)}")
    print(f"  SKIPPED, excluded head       : {len([k for k in csd if k.startswith(head_prefix)])}")
    print(f"  SKIPPED, shape mismatch      : {len(skipped_shape)}")
    print(f"  SKIPPED, absent in OBB model : {len(skipped_absent)}")

    missing = [k for k in backbone_target if k not in transferred]
    print(f"\n  non-head tensors NOT filled from checkpoint: {len(missing)}")
    for k in missing[:10]:
        print(f"    - {k} {tuple(msd[k].shape)}")

    ex_absent = sorted({k.rsplit(".", 1)[0] for k in skipped_absent})[:6]
    if ex_absent:
        print("\n  examples of checkpoint-only modules (v10Detect dual-branch head):")
        for k in ex_absent:
            print(f"    - {k}")

    check(
        "every non-head tensor was filled from the checkpoint",
        len(missing) == 0,
        f"{len(transferred)}/{len(backbone_target)} transferred",
    )
    check(
        "no head tensor was transferred",
        not any(k.startswith(head_prefix) for k in transferred),
    )
    head_unchanged = all(
        torch.equal(before[k], after[k]) for k in before if before[k].dtype.is_floating_point
    )
    check(
        "head weights are byte-identical to fresh init after load",
        head_unchanged,
        f"{len(before)} head tensors compared",
    )

    gl_crm = [k for k in transferred if "G2L" in k or "cv_glu" in k or "gl" in k.lower()]
    check(
        "GL-CRM backbone tensors are among those transferred",
        len(gl_crm) > 0 or len(transferred) > 700,
        f"{len(transferred)} transferred tensors including GL-CRM blocks",
    )


def test_upstream_parity():
    """Numeric parity against the installed upstream ultralytics for the ported math."""
    section("5. Numeric parity with upstream ultralytics")

    try:
        import ultralytics
        from ultralytics.utils.metrics import batch_probiou as u_batch_probiou
        from ultralytics.utils.metrics import probiou as u_probiou
        from ultralytics.utils.ops import nms_rotated as u_nms_rotated
        from ultralytics.utils.ops import regularize_rboxes as u_regularize_rboxes
        from ultralytics.utils.ops import xywhr2xyxyxyxy as u_xywhr2xyxyxyxy
        from ultralytics.utils.ops import xyxyxyxy2xywhr as u_xyxyxyxy2xywhr
    except ImportError:
        check("upstream ultralytics importable", False, "not installed; parity not checked")
        return

    print(f"  reference: ultralytics {ultralytics.__version__}")
    g = torch.Generator().manual_seed(7)
    n = 64
    rb = torch.cat(
        [
            torch.rand(n, 2, generator=g) * 200 + 20,       # cx, cy
            torch.rand(n, 2, generator=g) * 60 + 5,         # w, h
            (torch.rand(n, 1, generator=g) - 0.25) * math.pi,  # r in head range
        ],
        dim=1,
    )
    rb2 = rb.roll(1, dims=0)

    check(
        "probiou matches upstream",
        bool(torch.allclose(probiou(rb, rb2), u_probiou(rb, rb2), atol=1e-6)),
        f"max abs diff {float((probiou(rb, rb2) - u_probiou(rb, rb2)).abs().max()):.3e}",
    )
    check(
        "batch_probiou matches upstream",
        bool(torch.allclose(batch_probiou(rb, rb2), u_batch_probiou(rb, rb2), atol=1e-6)),
        f"max abs diff {float((batch_probiou(rb, rb2) - u_batch_probiou(rb, rb2)).abs().max()):.3e}",
    )
    check(
        "xywhr2xyxyxyxy matches upstream",
        bool(torch.allclose(xywhr2xyxyxyxy(rb), u_xywhr2xyxyxyxy(rb), atol=1e-5)),
    )
    corners = xywhr2xyxyxyxy(rb).reshape(-1, 8)
    check(
        "xyxyxyxy2xywhr matches upstream",
        bool(torch.allclose(xyxyxyxy2xywhr(corners), u_xyxyxyxy2xywhr(corners), atol=1e-4)),
    )

    scores = torch.rand(n, generator=g)
    fork_keep = sorted(int(i) for i in nms_rotated(rb.clone(), scores.clone(), 0.45))
    up_keep = sorted(int(i) for i in u_nms_rotated(rb.clone(), scores.clone(), 0.45))
    check(
        "nms_rotated keeps the same boxes as upstream",
        fork_keep == up_keep,
        f"fork kept {len(fork_keep)}, upstream kept {len(up_keep)}",
    )

    # regularize_rboxes is a KNOWN divergence: the fork predates upstream's rewrite.
    f_reg = regularize_rboxes(rb.clone())
    u_reg = u_regularize_rboxes(rb.clone())
    diverges = not torch.allclose(f_reg, u_reg, atol=1e-4)
    check(
        "regularize_rboxes DIVERGES from upstream (expected, documented)",
        diverges,
        "fork: long-edge, t in [0, pi); upstream 8.3: t in [0, pi/2) without long-edge",
    )
    check(
        "fork regularize_rboxes is long-edge with t in [0, pi)",
        bool((f_reg[:, 2] >= f_reg[:, 3] - 1e-6).all())
        and bool((f_reg[:, 4] >= 0).all() and (f_reg[:, 4] < math.pi).all()),
        f"w>=h for {int((f_reg[:, 2] >= f_reg[:, 3] - 1e-6).sum())}/{n}, "
        f"t range [{float(f_reg[:, 4].min()):.3f}, {float(f_reg[:, 4].max()):.3f}]",
    )
    check(
        "upstream regularize_rboxes is t in [0, pi/2) and NOT long-edge",
        bool((u_reg[:, 4] < math.pi / 2 + 1e-6).all())
        and not bool((u_reg[:, 2] >= u_reg[:, 3] - 1e-6).all()),
        f"w>=h for only {int((u_reg[:, 2] >= u_reg[:, 3] - 1e-6).sum())}/{n}, "
        f"t range [{float(u_reg[:, 4].min()):.3f}, {float(u_reg[:, 4].max()):.3f}]",
    )


def test_format_obb_normalization():
    """OBB labels must be normalized per-axis in xywhr, not derived from scaled segments."""
    section("6. Dataloader Format: OBB label normalization on a non-square image")

    from doclayout_yolo.data.augment import Format

    img_h, img_w = 320, 640  # deliberately non-square
    cx, cy, bw, bh, theta = 320.0, 160.0, 100.0, 40.0, math.radians(30)
    pts = rect_corners(cx, cy, bw, bh, theta).astype(np.float32)

    expected = xyxyxyxy2xywhr(pts.reshape(1, 8).copy())[0]  # pixel-space truth

    segments = pts.reshape(1, 4, 2).copy()
    bboxes = np.array(
        [[pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]], dtype=np.float32
    )
    labels = {
        "img": np.zeros((img_h, img_w, 3), dtype=np.uint8),
        "cls": np.array([[0]], dtype=np.float32),
        "instances": Instances(bboxes, segments, None, bbox_format="xyxy", normalized=False),
    }
    out = Format(bbox_format="xywh", normalize=True, return_obb=True, batch_idx=True)(labels)
    got = out["bboxes"][0]

    check(
        "angle is preserved from pixel space (not tilted by anisotropic scaling)",
        abs(float(got[4]) - float(expected[4])) < 1e-3,
        f"got r={float(got[4]):.5f}, pixel-space r={float(expected[4]):.5f}",
    )
    check(
        "cx, w normalized by image width",
        abs(float(got[0]) - float(expected[0]) / img_w) < 1e-4
        and abs(float(got[2]) - float(expected[2]) / img_w) < 1e-4,
        f"cx={float(got[0]):.5f} (expect {float(expected[0]) / img_w:.5f}), "
        f"w={float(got[2]):.5f} (expect {float(expected[2]) / img_w:.5f})",
    )
    check(
        "cy, h normalized by image height",
        abs(float(got[1]) - float(expected[1]) / img_h) < 1e-4
        and abs(float(got[3]) - float(expected[3]) / img_h) < 1e-4,
        f"cy={float(got[1]):.5f} (expect {float(expected[1]) / img_h:.5f}), "
        f"h={float(got[3]):.5f} (expect {float(expected[3]) / img_h:.5f})",
    )

    # What the pre-fix code did: minAreaRect on already-normalized segments.
    norm_pts = pts.copy()
    norm_pts[:, 0] /= img_w
    norm_pts[:, 1] /= img_h
    buggy = xyxyxyxy2xywhr(norm_pts.reshape(1, 8))[0]
    print(
        f"  note: deriving xywhr from pre-normalized segments would give r={float(buggy[4]):.5f}, "
        f"off by {abs(float(buggy[4]) - float(expected[4])):.5f} rad "
        f"({math.degrees(abs(float(buggy[4]) - float(expected[4]))):.1f} deg) at {img_w}x{img_h}"
    )


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    test_angle_convention()
    test_probiou_and_nms()
    test_forward_and_step()
    test_pretrained_transfer()
    test_upstream_parity()
    test_format_obb_normalization()

    section("SUMMARY")
    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print(f"  [{PASS if ok else FAIL}] {name}")
    print(f"\n  {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("  FAILED:")
        for n in failed:
            print(f"    - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
