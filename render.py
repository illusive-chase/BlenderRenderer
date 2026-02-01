import json
import os
from pathlib import Path
from subprocess import DEVNULL, call
from typing import Optional

import numpy as np

from utils import sphere_hammersley_sequence

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
    num_samples: int = 32,
    verbose: bool = False,
    seed: Optional[int] = None,
    light_seed: Optional[int] = None,
    fov_min: int = 10,
    fov_max: int = 90,
    radius_min: float = 0.35,
    radius_max: float = 0.6,
    save_mesh: bool = False,
    resolution: int = 518,
    min_pitch: Optional[float] = 0.0,
) -> None:
    assert file_path.exists() and file_path.suffix in ['.ply', '.glb', '.obj', '.blend']
    output_dir.mkdir(exist_ok=True, parents=True)
    assert fov_min <= fov_max
    assert radius_min <= radius_max
    assert resolution > 0 and num_views > 0

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
        pitchs.append(abs(p - min_pitch) + min_pitch)
    base_radius = np.random.rand() * (radius_max - radius_min) + radius_min
    radius_min = base_radius / np.sin(fov_max / 360 * np.pi)
    radius_max = base_radius / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (10000,))
    radius = 1 / np.sqrt(ks)
    fov = 2 * np.arcsin(base_radius / radius)
    radius = np.random.normal(1.0, 0.08, (10000,)) * radius
    
    # Randomly perturb camera placement and target
    center_perturbation = np.random.normal(0, 0.08, (10000, 3))
    cam_pos_perturbation = np.random.normal(0, 0.08, (10000, 3))
    
    views = []
    for i, (y, p, r, f) in enumerate(zip(yaws, pitchs, radius, fov)):
        views.append({
            'yaw': y,
            'pitch': p,
            'radius': r,
            'fov': f,
            'center': center_perturbation[i].tolist(),
            'pos_perturbation': cam_pos_perturbation[i].tolist()
        })

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', file_path.resolve().as_posix(),
        '--output_folder', output_dir.resolve().as_posix(),
        '--resolution', str(resolution),
        '--num_samples', str(num_samples),
    ]
    if light_seed is not None:
        args += ['--seed', str(light_seed)]
    if save_mesh:
        args += ['--save_mesh']
    if file_path.suffix == '.blend':
        args.insert(1, file_path.resolve().as_posix())

    call(args, stdout=None if verbose else DEVNULL)

if __name__ == '__main__':
    import tyro
    tyro.cli(render_cond)
