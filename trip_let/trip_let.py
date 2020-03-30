
import argparse
from itertools import product
import logging
from typing import List, Tuple, Iterable, NamedTuple

import numpy as np
from imageio import imread
import mcubes

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *_, **__):
        return x

__all__ = [
    'build_trip_let_volume'
]


class Transformation(NamedTuple):
    mirror: bool
    rotation: int


def all_transformations() -> Iterable[Transformation]:
    return (Transformation(*i) for i in product([False, True], range(4)))


def apply_transform(
        transformation: Transformation,
        array: np.ndarray
        ) -> np.ndarray:

    if transformation.mirror:
        array = array[:, ::-1]

    return np.rot90(array, transformation.rotation)


def build_trip_let_volume(
        proj0: np.ndarray,
        proj1: np.ndarray,
        proj2: np.ndarray
        ) -> np.ndarray:

    proj0 = proj0.astype(np.bool_)
    proj1 = proj1.astype(np.bool_)
    proj2 = proj2.astype(np.bool_)
    vol = np.logical_and(proj2[:, :, None], proj1[:, None, :])
    vol = np.logical_and(vol, proj0[None, :, :])
    return vol


def find_mistakes(
        volume: np.ndarray,
        projs: Iterable[np.ndarray]
        ) -> List[np.ndarray]:

    return [np.any(volume, axis=i) != proj_i
            for i, proj_i in enumerate(projs)]


def transform_and_build_volume(
        transforms: Iterable[Transformation],
        projs: Iterable[np.ndarray],
        ) -> Tuple[np.ndarray, List[np.ndarray], Iterable[Transformation]]:

    images = [apply_transform(i, j) for i, j in zip(transforms, projs)]
    volume = build_trip_let_volume(*images)
    mistakes = find_mistakes(volume, images)
    return volume, mistakes, transforms


def imread_nochannels(filename: str) -> np.ndarray:

    image = imread(filename)

    if image.ndim > 2:
        image = image[..., 0]

    return image


def parse_args():

    parser = argparse.ArgumentParser("trip-let")
    parser.add_argument('-x', type=str, required=True)
    parser.add_argument('-y', type=str, required=True)
    parser.add_argument('-z', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)

    # Transformations
    parser.add_argument('--rotx', type=int, default=0)
    parser.add_argument('--roty', type=int, default=0)
    parser.add_argument('--rotz', type=int, default=0)
    parser.add_argument('--mirrorx', action='store_true')
    parser.add_argument('--mirrory', action='store_true')
    parser.add_argument('--mirrorz', action='store_true')
    parser.add_argument('--find-best-transform', action='store_true')

    # Output mistakes
    # parser.add_argument('--mistakes', action='store_true')

    # Output volume
    # parser.add_argument('--output-volume', type=str)

    parser.add_argument('--smoothing',
                        choices=['no', 'auto', 'constrained', 'gaussian'],
                        default='gaussian')
    parser.add_argument('--no-center', action='store_true')

    return parser.parse_args()


def main():

    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    filenames = (args.x, args.y, args.z)

    logging.info("Loading images...")
    images = [imread_nochannels(i) > 128 for i in filenames]

    if not args.find_best_transform:
        # Use the user-given transformation
        transf_x = Transformation(args.mirrorx, args.rotx)
        transf_y = Transformation(args.mirrory, args.roty)
        transf_z = Transformation(args.mirrorz, args.rotz)
        transforms = (transf_x, transf_y, transf_z)

        logging.info("Building trip-let volume...")
        volume, mistakes, _ = transform_and_build_volume(transforms, images)
    else:
        # Iterate over all possible transformations and pick the one that
        # minimizes the number of mistakes.
        logging.info("Finding the best transformation...")

        all_transforms3 = product(all_transformations(), repeat=3)
        all_transforms3 = tqdm(list(all_transforms3))

        volume, mistakes, best_transformation = min(
            (transform_and_build_volume(t, images) for t in all_transforms3),
            # key function counts the number of mistakes
            key=lambda x: sum(np.sum(i) for i in x[1])
        )

        logging.info("Best transformation: %s", best_transformation)

    logging.info("%d unmet constraints found",
                 sum(np.sum(i) for i in mistakes))

    # Smoothing
    isovalue = 0.0
    if args.smoothing == 'auto':
        volume = mcubes.smooth(volume)
    elif args.smoothing == 'constrained':
        volume = mcubes.smooth_constrained(volume, max_iters=500)
    elif args.smoothing == 'gaussian':
        volume = mcubes.smooth_gaussian(volume)
    else:
        isovalue = 0.5

    # Marching cubes
    logging.info("Marching cubes...")
    vertices, triangles = mcubes.marching_cubes(volume, isovalue)

    # Center the mesh
    if not args.no_center:
        vertices = vertices - vertices.mean(axis=0)[None]

    # Export the mesh
    logging.info("Exporting...")
    mcubes.export_obj(vertices, triangles, args.o)
    logging.info("Done.")


if __name__ == '__main__':
    main()
