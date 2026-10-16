#!/usr/bin/env python
"""
v16 anisotropic mesh builder (gmsh): true metric/boundary-layer remeshing.

Full-domain triangulation with the AIRFOIL CARVED OUT and ANISOTROPIC wall-stretched
boundary-layer triangles hugging the airfoil (thin in the wall-normal direction), grading
to a coarse far field. Unlike the v13/v14 MeshLab remesh (isotropic-adaptive), this is a
genuine anisotropic BL mesh via gmsh's BoundaryLayer field + a distance-graded size field.

Robustness: the airfoil outline is ordered into a GUARANTEED-SIMPLE (non-self-intersecting)
polygon via a camber-line upper/lower split (`order_airfoil_camber`) — the earlier greedy
nearest-neighbour ordering hopped between the close upper/lower surfaces of thin airfoils and
produced self-intersecting loops, on which gmsh silently emitted zero triangles (~21% of the
dataset). The outline is then arc-length resampled to `n_poly` points (keeps it simple, and
matches the approved ~5k-vertex / ~10k-triangle resolution). If the BoundaryLayer field still
fails for a pathological geometry, `build_v16_mesh` falls back down a bl_thick ladder and,
last resort, to a graded (near-wall-refined, non-BL) mesh, so every sample yields a valid mesh.

build_v16_mesh(pos, sdf, ...) -> (V[:,2], F, wall_vertex_mask)
interpolate_fields reused from v13_mesh.
"""
import numpy as np
from scipy.spatial import cKDTree
import multiprocessing as _mp
import gmsh
from shapely.geometry import Polygon


def order_airfoil_camber(P, nbin=60, eps=1e-6):
    """Order airfoil boundary points into a simple closed polygon.

    Chord frame: LE = min-x point, TE = max-x point. Classify each point as upper/lower
    by its offset from the local CAMBER line (per-x-station midpoint of the offsets),
    which is robust to camber/undercamber where a straight chord line misclassifies. Then
    walk upper LE->TE and lower TE->LE. Guarantees an x-monotone (hence simple) loop for
    single-valued airfoils. Drops near-duplicate consecutive points.
    """
    P = np.asarray(P, dtype="float64")
    i_le = int(np.argmin(P[:, 0])); i_te = int(np.argmax(P[:, 0]))
    le = P[i_le]; te = P[i_te]
    d = te - le; d = d / (np.linalg.norm(d) + 1e-12)
    nrm = np.array([-d[1], d[0]])
    t = (P - le) @ d          # along-chord coordinate
    s = (P - le) @ nrm        # normal offset
    edges = np.linspace(t.min(), t.max(), nbin + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    camb = np.full(nbin, np.nan)
    for b in range(nbin):
        m = (t >= edges[b]) & (t <= edges[b + 1])
        if m.any():
            camb[b] = 0.5 * (s[m].min() + s[m].max())
    ok = ~np.isnan(camb)
    camb_t = np.interp(t, mid[ok], camb[ok])
    up = s >= camb_t
    ui = np.where(up)[0]; li = np.where(~up)[0]
    order = np.concatenate([ui[np.argsort(t[ui])], li[np.argsort(-t[li])]])
    poly = P[order]
    keep = [poly[0]]
    for pt in poly[1:]:
        if np.linalg.norm(pt - keep[-1]) > eps:
            keep.append(pt)
    poly = np.array(keep)
    if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < eps:
        poly = poly[:-1]
    return poly


def _repair_polygon(poly):
    """Guarantee a simple (non-self-intersecting) polygon via Shapely buffer(0).

    The camber-line ordering removes most self-intersections, but a few airfoils with
    blunt/crossed trailing edges still order into a slightly self-intersecting loop, on
    which gmsh emits zero triangles (even for the graded fallback). buffer(0) snaps such
    loops to a valid polygon; if it splits into several pieces we keep the largest. Only
    the tiny crossed TE region is affected, so the airfoil shape is preserved.
    """
    p = Polygon(poly)
    if p.is_valid:
        return poly
    p = p.buffer(0)
    if p.is_empty:
        return poly
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda g: g.area)
    ext = np.asarray(p.exterior.coords)[:-1]   # drop closing duplicate
    return ext if len(ext) >= 8 else poly


def _arclen_resample(poly, n):
    """Uniform arc-length resampling of a closed polygon to n points (keeps it simple)."""
    if len(poly) <= n:
        return poly
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tot = s[-1]
    targ = np.linspace(0.0, tot, n, endpoint=False)
    xi = np.interp(targ, s, closed[:, 0])
    yi = np.interp(targ, s, closed[:, 1])
    return np.column_stack([xi, yi])


def _mesh_once(poly, bbox, hwall, hfar, bl_thick, bl_ratio, use_bl):
    """Single gmsh attempt. Returns (V, F) or (None, None) if no triangles."""
    x0, y0, x1, y1 = bbox
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("af")
    g = gmsh.model.geo
    c = [g.addPoint(x0, y0, 0, hfar), g.addPoint(x1, y0, 0, hfar),
         g.addPoint(x1, y1, 0, hfar), g.addPoint(x0, y1, 0, hfar)]
    outer = [g.addLine(c[i], c[(i + 1) % 4]) for i in range(4)]
    outer_loop = g.addCurveLoop(outer)
    ap = [g.addPoint(px, py, 0, hwall * 30) for px, py in poly]
    al = [g.addLine(ap[i], ap[(i + 1) % len(ap)]) for i in range(len(ap))]
    aloop = g.addCurveLoop(al)
    g.addPlaneSurface([outer_loop, aloop])
    g.synchronize()

    fld = gmsh.model.mesh.field
    if use_bl:
        bl = fld.add("BoundaryLayer")
        fld.setNumbers(bl, "CurvesList", al)
        fld.setNumber(bl, "Size", hwall)
        fld.setNumber(bl, "SizeFar", hfar)
        fld.setNumber(bl, "Thickness", bl_thick)
        fld.setNumber(bl, "Ratio", bl_ratio)
        fld.setNumber(bl, "Quads", 0)
        fld.setAsBoundaryLayer(bl)
    dist = fld.add("Distance"); fld.setNumbers(dist, "CurvesList", al); fld.setNumber(dist, "Sampling", 400)
    thr = fld.add("Threshold")
    fld.setNumber(thr, "InField", dist)
    fld.setNumber(thr, "SizeMin", hwall * 12); fld.setNumber(thr, "SizeMax", hfar)
    fld.setNumber(thr, "DistMin", bl_thick);   fld.setNumber(thr, "DistMax", 1.2)
    fld.setAsBackgroundMesh(thr)
    for opt in ["Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"]:
        gmsh.option.setNumber(opt, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)   # Delaunay

    V = F = None
    try:
        gmsh.model.mesh.generate(2)
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        Vv = ncoord.reshape(-1, 3)[:, :2]
        tag2i = {int(t): i for i, t in enumerate(ntags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        for et, en in zip(etypes, enodes):
            if et == 2:  # 3-node triangle
                F = np.array([[tag2i[int(t)] for t in row] for row in en.reshape(-1, 3)], dtype="int64")
                V = Vv
    except Exception:
        V = F = None
    gmsh.finalize()
    if F is None or len(F) == 0:
        return None, None
    return V, F


# ── subprocess-guarded meshing ────────────────────────────────────────────────
# gmsh's BoundaryLayer field can spin indefinitely in its internal 1D-edge-recovery
# loop when the offset layers collide near a sharp trailing edge (inherent to BL
# meshing on airfoils, not fixable by ordering). We therefore run every gmsh attempt
# in a short-lived child process with a hard wall-clock timeout: if it hangs, the
# child is killed and we fall back, so no sample can ever freeze the batch.

def _mesh_worker(q, poly, bbox, hwall, hfar, bl_thick, bl_ratio, use_bl):
    try:
        V, F = _mesh_once(poly, bbox, hwall, hfar, bl_thick, bl_ratio, use_bl)
    except Exception:
        V, F = None, None
    q.put((None if V is None else V.astype("float32"),
           None if F is None else F.astype("int64")))


def _mesh_guarded(poly, bbox, hwall, hfar, bl_thick, bl_ratio, use_bl, timeout):
    """Run one gmsh attempt in a child process; kill it if it exceeds `timeout` s.

    Returns (V, F, timed_out): (None, None, True) on hang, (None, None, False) on a
    clean no-triangle failure, or (V, F, False) on success.
    """
    # spawn (not fork): the parent has imported torch/gmsh which start background
    # threads; forking a multithreaded process gives the child inherited locked
    # mutexes and it deadlocks in gmsh.initialize(). spawn starts a fresh interpreter
    # (imports only this lightweight numpy+scipy+gmsh module), sidestepping that.
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_mesh_worker,
                    args=(q, poly, bbox, hwall, hfar, bl_thick, bl_ratio, use_bl))
    p.start()
    timed_out = False
    try:
        V, F = q.get(timeout=timeout)
    except Exception:                                    # queue.Empty -> child hung
        V, F = None, None
        timed_out = True
    p.join(1.0)
    if p.is_alive():
        p.terminate(); p.join(1.0)
        if p.is_alive():
            p.kill(); p.join(1.0)
    q.close()
    return V, F, timed_out


def build_v16_mesh(pos, sdf, hwall=0.0025, hfar=0.13, bl_thick=0.05, bl_ratio=1.25,
                   n_poly=220, min_tris=2000, bl_timeout=10.0, graded_timeout=15.0):
    """Anisotropic BL mesh: thin wall-normal layers on the airfoil, coarse far field.

    Robust pipeline: simple-polygon airfoil ordering + arc-length resample, then a BL
    thickness ladder (each attempt subprocess-guarded with a hard timeout so a spinning
    gmsh cannot hang), and finally a graded (non-BL, still wall-refined) fallback so no
    sample fails. Returns (V, F, wall_mask, info) where info records the method used
    ('bl' with the accepted thickness, or 'graded') for dataset provenance.
    """
    wall = np.where(sdf == 0.0)[0]
    poly = order_airfoil_camber(pos[wall])
    poly = _repair_polygon(poly)               # snap any residual self-intersection to a valid loop
    poly = _arclen_resample(poly, n_poly)
    bbox = (float(pos[:, 0].min()), float(pos[:, 1].min()),
            float(pos[:, 0].max()), float(pos[:, 1].max()))

    V = F = None
    used_bt = None
    for bt in (bl_thick, bl_thick * 0.6, bl_thick * 0.36):    # smaller BL fits thin foils
        Vc, Fc, _ = _mesh_guarded(poly, bbox, hwall, hfar, bt, bl_ratio,
                                  use_bl=True, timeout=bl_timeout)
        if Fc is not None and len(Fc) >= min_tris:
            V, F, used_bt = Vc, Fc, bt
            break
    if F is None:                                             # graded fallback (still wall-refined)
        V, F, _ = _mesh_guarded(poly, bbox, hwall, hfar, bl_thick, bl_ratio,
                                use_bl=False, timeout=graded_timeout)
        info = {"method": "graded", "bl_thick": None}
    else:
        info = {"method": "bl", "bl_thick": float(used_bt)}
    if F is None:
        raise RuntimeError("gmsh produced no triangles even for the graded fallback")

    wall_v = cKDTree(poly).query(V)[0] < hwall * 6
    return V.astype("float32"), F, wall_v, info
