"""
interpolate_fields — carried over from the parent research repo's v13_mesh.py, trimmed to just
this one function. The rest of that file (build_v13_mesh, an earlier isotropic MeshLab remesher
superseded by v16_mesh.py's anisotropic boundary-layer one) is not used by this pipeline and
pulled in a dependency (`dataset_airfrans_v12._order_airfoil_loop`) this deployment doesn't
otherwise need — dropped rather than vendored for no reason.
"""
import numpy as np
from scipy.interpolate import griddata


def interpolate_fields(pos, vals, V):
    """vals: (N, C) per raw point -> (M, C) at mesh vertices V (linear + nearest fill)."""
    out = griddata(pos, vals, V, method="linear")
    if out.ndim == 1:
        out = out[:, None]
    nan = np.isnan(out).any(axis=1)
    if nan.any():
        fill = griddata(pos, vals, V[nan], method="nearest")
        out[nan] = fill if fill.ndim == 2 else fill[:, None]
    return out.astype("float32")
