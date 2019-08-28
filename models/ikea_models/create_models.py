#!/usr/bin/env python3
import os
import os.path as osp
from pathlib import Path
import subprocess

import numpy as np
from yaml import load, dump
from tvtk.api import tvtk


def find_files(topdir, extensions=[".mtl", ".obj"]):
    for dirpath, dirnames, filenames in os.walk(topdir):
        if all(any(f.endswith(ext) for f in filenames) for ext in extensions):
            objfile = [f for f in filenames if f.endswith(".obj")]
            yield osp.join(dirpath, objfile[0])


def tvtk_reader(objfile):
    obj_reader = tvtk.OBJReader(file_name=objfile)
    obj_reader.update()
    return obj_reader.output


def tvtk_get_points(polydata):
    return polydata.points.to_array()


def tvtk_get_center_of_mass(polydata):
    com_filter = tvtk.CenterOfMass(use_scalars_as_weights=False)
    com_filter.set_input_data(polydata)
    com_filter.update()
    return com_filter.center


def mesh_shift_scale_normalize_max(objfile, maxdim=1):
    polydata = tvtk_reader(objfile)
    points = tvtk_get_points(polydata)
    mins = np.quantile(points, 0.05, axis=0)#  points.min(axis=0)
    maxs = np.quantile(points, 0.95, axis=0)# points.max(axis=0)
    # center = tvtk_get_center_of_mass(polydata)
    dims = (maxs - mins)
    mesh_maxdim = dims.max()
    rescale_factor = np.ones(3) * maxdim / mesh_maxdim
    rescaled_dims = dims * rescale_factor
    return rescaled_dims.tolist(), rescale_factor.tolist()


def main(topdir="ikea_models/meshes",
         template_fmt="ikea_models/{file}.erb",
         generated_files=["model.config", "model.sdf"],
         desired_max_dim=dict(
             bed=3,
             bookcase=3,
             chair=1,
             desk=1.5,
             table=1.5,
             sofa=2.0,
             wardrobe=3)):
    for objfile in find_files(topdir):
        objfileparts = objfile.split(osp.sep)
        ikea_names = [fp for fp in objfileparts if fp.startswith("IKEA_")]
        _, ikea_type, *ikea_subtype = ikea_names[0].split("_")
        ikea_model_dir = osp.join("ikea_models", ikea_names[0])
        objdims, objscale = mesh_shift_scale_normalize_max(
            objfile,
            maxdim=desired_max_dim[ikea_type])
        Path(ikea_model_dir).mkdir(parents=True, exist_ok=True)
        for f in generated_files:
            p = subprocess.Popen(["erb", template_fmt.format(file=f)],
                                 stdout=open(osp.join(ikea_model_dir, f), "w"),
                                 env=dict(IKEA_OBJ_FILENAME=objfile,
                                          IKEA_OBJ_SCALE=dump(objscale),
                                          IKEA_OBJ_DIMS=dump(objdims)))
            p.wait(timeout=30)


if __name__ == '__main__':
    main()
