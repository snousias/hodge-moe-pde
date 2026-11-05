from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import potpourri3d as pp3d

from torch.utils.data import Dataset

from .transforms import apply_vertex_transforms
from .transforms_o3d import apply_mesh_transforms
from .utils import sparse_scipy_to_torch

try:
    from mesh_sim.mesh import MeshSignedIncidenceMatrices
    from mesh_sim.utils.open3d import build_o3d_mesh
    from mesh_sim.utils_ml import extract_mesh_features
except ImportError:
    raise ImportError("Need to install mesh-sim library first.")


class MeshPDEDataset(Dataset):
    """
    Dataset for PDE regression on triangular meshes.

    Each sample is a triangular mesh (domain) paired with a scalar field
    (e.g., pressure) defined at each mesh vertex.  The dataset returns the
    standard HodgeFormer mesh-feature tuple together with the target field.

    Parameters
    ----------
    mesh_paths : List[str]
        Paths to mesh files (.ply / .obj / .off).
    press_paths : List[str]
        Paths to ``press.npy`` arrays.  Each array must have length ≥ nv;
        only the first ``nv`` values are used (extra CFD ghost nodes are
        silently discarded).
    transforms : List[str] | None
        Geometric augmentation tags applied to the mesh (e.g.
        ``["vertex_center_scale", "rotate", "jitter"]``).
    press_mean : float | None
        Pre-computed training-set mean for pressure normalisation.
        If ``None``, the raw pressure values are returned.
    press_std : float | None
        Pre-computed training-set std for pressure normalisation.
    extract_kw : Dict
        Keyword arguments forwarded to ``extract_mesh_features``.
    """

    def __init__(
        self,
        mesh_paths: List[str],
        press_paths: List[str],
        transforms: Optional[List[str]] = None,
        press_mean: Optional[float] = None,
        press_std: Optional[float] = None,
        extract_kw: Dict = {"mode": "bigbird"},
    ):
        super().__init__()

        assert len(mesh_paths) == len(press_paths), (
            "mesh_paths and press_paths must have the same length"
        )

        self.mesh_paths = mesh_paths
        self.press_paths = press_paths
        self.transforms = transforms
        self.press_mean = press_mean
        self.press_std = press_std
        self.extract_kw = extract_kw
        self.rs = np.random.RandomState(seed=123)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        tensors : tuple
            ``(x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx)``
        press : torch.FloatTensor, shape ``(nv,)``
            Normalised pressure at each vertex.
        """
        vertices, faces = pp3d.read_mesh(self.mesh_paths[idx])

        mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        if self.transforms:
            apply_mesh_transforms(mesh_o3d, self.transforms, rs=self.rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(
            mesh_o3d=mesh_o3d, uid=self.mesh_paths[idx]
        )

        x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx = extract_mesh_features(
            mesh, **self.extract_kw
        )

        nv = x_v.shape[0]

        # Load pressure; silently drop the last (3682 - nv) ghost-node values
        press = np.load(self.press_paths[idx]).astype(np.float32)[:nv]

        if self.press_mean is not None and self.press_std is not None:
            press = (press - self.press_mean) / (self.press_std + 1e-8)

        return (
            (
                torch.FloatTensor(x_v),
                torch.FloatTensor(x_e),
                torch.FloatTensor(x_f),
                sparse_scipy_to_torch(d_0, transpose=True),
                sparse_scipy_to_torch(d_1, transpose=True),
                torch.LongTensor(v_idx),
                torch.LongTensor(e_idx),
                torch.LongTensor(f_idx),
            ),
            torch.FloatTensor(press),
        )

    # ------------------------------------------------------------------

    @classmethod
    def compute_press_stats(
        cls, press_paths: List[str], nv: int
    ) -> Tuple[float, float]:
        """
        Compute global mean and std of pressure across a list of samples.
        Only the first ``nv`` values of each press.npy are used.
        """
        all_vals = np.concatenate(
            [np.load(p).astype(np.float32)[:nv] for p in press_paths]
        )
        return float(all_vals.mean()), float(all_vals.std())
