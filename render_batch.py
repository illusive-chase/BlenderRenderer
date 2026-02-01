from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from render import _install_blender, render_cond


def _render_wrapper(
    file_path: Path,
    *,
    root_folder: Path,
    output_root: Path,
    **kwargs,
) -> None:
    try:
        # Calculate relative path to preserve directory structure
        rel_path = file_path.relative_to(root_folder)
        # Create output directory for this specific object
        # e.g. root/category/object.glb -> output_root/category/object/
        target_dir = output_root / rel_path.with_suffix('')
        
        render_cond(
            file_path,
            output_dir=target_dir,
            **kwargs,
        )
    except Exception as e:
        print(f"Error rendering {file_path}: {e}")


def render_cond_batch(
    folder: Path,
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
    num_workers: int = 32,
) -> None:
    # Ensure blender is installed before starting processes
    _install_blender()

    # Find all supported files
    extensions = ['.glb', '.obj', '.ply', '.blend']
    data = []
    for ext in extensions:
        data.extend(list(folder.glob(f"**/*{ext}")))
    
    print(f'Found {len(data)} objects')
    
    worker = partial(
        _render_wrapper,
        root_folder=folder,
        output_root=output_dir,
        num_views=num_views,
        num_samples=num_samples,
        verbose=verbose,
        seed=seed,
        light_seed=light_seed,
        fov_min=fov_min,
        fov_max=fov_max,
        radius_min=radius_min,
        radius_max=radius_max,
        save_mesh=save_mesh,
        resolution=resolution,
        min_pitch=min_pitch,
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(worker, data), total=len(data), dynamic_ncols=True))

if __name__ == '__main__':
    import tyro
    tyro.cli(render_cond_batch)
