import os
import json
import copy
import sys
import importlib
import argparse
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from pathlib import Path
from utils import sphere_hammersley_sequence
import tyro
from typing import Optional

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = os.path.dirname(__file__)
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender() -> None:
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxxf86vm1 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def render_cond(
    file_path: Path,
    *,
    output_dir: Path,
    num_views: int,
    verbose: bool = False,
    seed: Optional[int] = None,
    fov_min: int = 10,
    fov_max: int = 70,
    base_radius: float = np.sqrt(3) / 2,
    save_mesh: bool = False,
    resolution: int = 1024,
) -> None:
    assert file_path.exists() and file_path.suffix in ['.ply', '.glb', '.obj', '.blend']
    output_dir.mkdir(exist_ok=True, parents=True)
    assert fov_max <= fov_max
    assert resolution > 0 and base_radius > 0 and num_views > 0

    if seed is not None:
        np.random.seed(seed)

    print('Checking blender...', flush=True)
    _install_blender()
    print('Rendering...', flush=True)

    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius_min = base_radius / np.sin(fov_max / 360 * np.pi)
    radius_max = base_radius / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(base_radius / r) for r in radius]
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', file_path.resolve().as_posix(),
        '--output_folder', output_dir.resolve().as_posix(),
        '--resolution', resolution,
    ]
    if save_mesh:
        args += ['--save_mesh']
    if file_path.suffix == '.blend':
        args.insert(1, file_path.resolve().as_posix())

    call(args, stdout=None if verbose else DEVNULL)

if __name__ == '__main__':
    tyro.cli(render_cond)
