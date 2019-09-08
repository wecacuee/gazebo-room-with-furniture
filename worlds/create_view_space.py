#!/usr/bin/env python3
import os.path as osp
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from keyword2cmdline import command


def relpath(fname,
            reldir=osp.dirname(__file__) or "."):
    return osp.join(reldir, fname)


def create_view_space(mins, maxs, incs):
    grid = np.mgrid[mins[0]:maxs[0]:incs[0],
                    mins[1]:maxs[1]:incs[1],
                    mins[2]:maxs[2]:incs[2],
                    mins[3]:maxs[3]:incs[3],
                    mins[4]:maxs[4]:incs[4],
                    mins[5]:maxs[5]:incs[5]]
    print(grid.shape)
    return grid


def write_view_space(dest_file, view_grid):
    points = view_grid.T.reshape(-1, 6)
    npts = points.shape[0]
    dest_dir = osp.dirname(dest_file)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with open(dest_file, "w") as f:
        f.write("%d\n" % npts)
        quat = Rotation.from_euler('xyz', points[:, 3:]).as_quat()
        r, theta, z = points[:, 0:1], points[:, 1:2], points[:, 2:3]
        xyz_quat = np.hstack(
            (r*np.cos(theta), r*np.sin(theta), z, quat))
        np.savetxt(f, xyz_quat)


@command
def create_view_space_files(view_space_file_fmt="worlds/world%d/view_space.txt",
                            mins=[ 5,   -np.pi, 2.0, -np.pi/6, -np.pi/6,  -np.pi],
                            maxs=[10,    np.pi, 2.5,  np.pi/6,  np.pi/6,   np.pi],
                            incs=[ 1, np.pi/10, 0.5,  np.pi/6,  np.pi/6, 2*np.pi/9]):
    """
    Creates a view_space file compatible with
    ig_active_reconstruction/view_space.cpp:LoadFromFile
    """
    for i in range(100):
        write_view_space(
            relpath(view_space_file_fmt % i),
            create_view_space(mins, maxs, incs))


if __name__ == '__main__':
    create_view_space_files()

