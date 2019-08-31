#!/usr/bin/env python3
import os
import os.path as osp
from pathlib import Path
import subprocess

import numpy as np
from yaml import load, dump
import vtk
from mayavi_array_handler import vtk2array

def find_files(topdir, extensions=[".mtl", ".obj"]):
    for dirpath, dirnames, filenames in os.walk(topdir):
        if all(any(f.endswith(ext) for f in filenames) for ext in extensions):
            objfile = [f for f in filenames if f.endswith(".obj")]
            yield osp.join(dirpath, objfile[0])


def vtk_reader(objfile):
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(objfile)
    obj_reader.Update()
    return obj_reader.GetOutput()


def vtk_get_points(polydata):
    return vtk2array(polydata.GetPoints().GetData())


def vtk_get_center_of_mass(polydata):
    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetUseScalarsAsWeights(False)
    com_filter.SetInputData(polydata)
    com_filter.Update()
    return np.array(com_filter.GetCenter())


def filepath_add_suffix(filepath, suffix, sep="-"):
    fileprefix, ext = osp.splitext(filepath)
    return sep.join((fileprefix, suffix)) + ext


def vtk_writer(polydata, objfile):
    assert polydata
    from packaging import version
    assert (version.parse(vtk.vtkVersion().GetVTKVersion()) >=
            version.parse("8.2.0")), "OBJWriter is not available before 8.2.0"
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetFileName(objfile)
    obj_writer.SetInputData(polydata)
    obj_writer.Update()


def vtk_shift_polydata(polydata, shift):
    translation = vtk.vtkTransform()
    translation.Translate(*shift)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(translation)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()
    return transform_filter.GetOutput()


def vtk_get_volume(polydata):
    massprop = vtk.vtkMassProperties()
    massprop.SetInputData(polydata)
    massprop.Update()
    return massprop.GetVolume()


def write_corresponding_mtl_file(objfile, centered_objfile):
    objbaseprefix, _ = osp.splitext(osp.basename(objfile))
    centered_objfileprefix, _ = osp.splitext(centered_objfile)

    try:
        os.symlink(objbaseprefix + ".mtl", centered_objfileprefix + ".mtl")
    except FileExistsError:
        pass


def parse_sketchup_file_units(objfile):
    for line in open(objfile):
        if line.startswith("# File units = "):
            _, unitstr = line.split("=")
            unitstr = unitstr.strip()
            return unitstr


def unitstr2rescale_factor(unitstr,
                           conversions=dict(meters=1,
                                            centimeters=0.01,
                                            millimeters=0.001,
                                            inches=0.0254)):
    return conversions[unitstr]


def mesh_shift_scale_from_comments(objfile):
    polydata = vtk_reader(objfile)
    points = vtk_get_points(polydata)
    mins = np.quantile(points, 0.05, axis=0)#  points.min(axis=0)
    maxs = np.quantile(points, 0.95, axis=0)# points.max(axis=0)
    dims = (maxs - mins)

    volume = vtk_get_volume(polydata)

    centered_objfile = filepath_add_suffix(objfile, "centered")
    rescale_factor = np.ones(3) * unitstr2rescale_factor(
        parse_sketchup_file_units(objfile))
    rescaled_dims = dims * rescale_factor
    return rescaled_dims.tolist(), rescale_factor.tolist(), volume

def mesh_shift_scale_normalize_max(objfile,
                                   desired_max_dim=dict(
                                       bed=3,
                                       bookcase=3,
                                       chair=1,
                                       desk=1.5,
                                       table=1.5,
                                       sofa=2.0,
                                       wardrobe=3)):
    _, ikea_type, *ikea_subtype = ikea_names[0].split("_")
    maxdim = desired_max_dim[ikea_type]
    polydata = vtk_reader(objfile)
    points = vtk_get_points(polydata)
    mins = np.quantile(points, 0.05, axis=0)#  points.min(axis=0)
    maxs = np.quantile(points, 0.95, axis=0)# points.max(axis=0)
    center = vtk_get_center_of_mass(polydata)
    centered_objfile = filepath_add_suffix(objfile, "centered")
    ### Does not work because vtkOBJReader does not read textures
    ### pywavefront does but it does not write objects
    ### Mapping pywavefront textures to a vtkActor + PolyData will be hard.
    ### vtkOBJImporter and vtkOBJExporter read obj files for rendering
    ### but do not export to polydata where vertex manipulation can be done.
    # shifted_polydata = vtk_shift_polydata(polydata, -center)
    # vtk_writer(shifted_polydata, centered_objfile)
    # write_corresponding_mtl_file(objfile, centered_objfile)
    dims = (maxs - mins)
    mesh_maxdim = dims.max()
    rescale_factor = np.ones(3) * maxdim / mesh_maxdim
    rescaled_dims = dims * rescale_factor
    return rescaled_dims.tolist(), rescale_factor.tolist(), 0


def main(topdir="ikea_models/meshes",
         template_fmt="ikea_models/{file}.erb",
         generated_files=["model.config", "model.sdf"]):
    for objfile in find_files(topdir):
        objfileparts = objfile.split(osp.sep)
        ikea_names = [fp for fp in objfileparts if fp.startswith("IKEA_")]
        ikea_model_dir = osp.join("ikea_models", ikea_names[0])
        objdims, objscale, objvolume = mesh_shift_scale_from_comments(
            objfile)
        Path(ikea_model_dir).mkdir(parents=True, exist_ok=True)
        for f in generated_files:
            p = subprocess.Popen(["erb", template_fmt.format(file=f)],
                                 stdout=open(osp.join(ikea_model_dir, f), "w"),
                                 env=dict(IKEA_OBJ_FILENAME=objfile,
                                          IKEA_OBJ_SCALE=dump(objscale),
                                          IKEA_OBJ_DIMS=dump(objdims),
                                          IKEA_OBJ_VOLUME=dump(objvolume)))
            p.wait(timeout=30)


if __name__ == '__main__':
    main()
