from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from moc.core.models.network import Network
from moc.core.build.discretize import DiscretizationResult, PipeMesh
from moc.core.postprocess.summary import build_pipe_ranges_accumulated, PipeRange


@dataclass(frozen=True)
class GeometryProfile:
    s_m: np.ndarray   # [npts]
    z_m: np.ndarray   # [npts]


def build_z_profile_linear(network: Network, disc: DiscretizationResult) -> GeometryProfile:
    """
    Construye z(s) sobre la malla global del solver, asumiendo z lineal por tubo:
      z(s) = interp( z_from -> z_to ) en cada tubo
    """
    ranges: List[PipeRange] = build_pipe_ranges_accumulated(network, disc)

    # npts = last i_end + 1
    npts = ranges[-1].i_end + 1
    s = np.zeros(npts, dtype=float)
    z = np.zeros(npts, dtype=float)

    for r in ranges:
        p = network.pipes[r.pipe_uid]
        n_from = network.nodes[p.node_from]
        n_to = network.nodes[p.node_to]

        z0 = float(n_from.z)
        z1 = float(n_to.z)

        mesh: PipeMesh = disc.meshes_by_pipe[r.pipe_uid]
        x_local = np.array(mesh.x, dtype=float)   # 0..L
        L = float(p.length)

        # acumulado
        s_seg = r.s_start_m + x_local

        # interpolación lineal de z
        if L == 0:
            z_seg = np.full_like(x_local, z0)
        else:
            w = x_local / L
            z_seg = (1.0 - w) * z0 + w * z1

        # asignación global
        i0, i1 = r.i_start, r.i_end
        expected = i1 - i0 + 1
        if expected != len(s_seg):
            raise ValueError("Inconsistencia de malla al construir z(s).")

        s[i0:i1 + 1] = s_seg
        z[i0:i1 + 1] = z_seg

    return GeometryProfile(s_m=s, z_m=z)
