#!/usr/bin/env python3
from functools import wraps
from inspect import getargspec
from fcntl import lockf, LOCK_UN, LOCK_EX
from contextlib import contextmanager
import os.path as osp
import subprocess
from collections import namedtuple
import random
from jinja2 import Template


import numpy as np


def relpath(fname,
            reldir=osp.dirname(__file__) or "."):
    return osp.join(reldir, fname)


def class_name_from_idx(
        idx):
    # self.type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
    from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
    mapping = SunrgbdDatasetConfig().class2type
    return mapping[idx]

ALL_GOOD_POSES_STR = """
        6.37731 -16.6967 0.292783 0 -0 1.56052
        7.34128 1.42961 0.164696 0.002975 0.004672 -0.002476
        3.64071 7.8499 -0.002945 -0.000138 -0 -0
        8.48345 0.283019 -0.001321 -0.000133 0 0
        8.56815 4.50743 0 -0.000226 0 0
        -8.67513 -8.89397 0.284081 -0.008142 0.070161 -3.105091
        4.66112 8.06005 -0 -0 -0 -0
        7.87416 7.83227 -0.002873 -0.000127 0 0
        -0.727502 8.02459 0 0 -0 0
        7.3625 -2.93731 0.048715 -4.6e-05 0 0
        -9.39874 -11.4083 -0.048035 -8.1e-05 0.009141 0.002019
        -4.35584 -7.88196 -3e-06 -0.010162 -1.2e-05 0.00027
        -2.40996 -8.82201 -0.000605 0.000125 0.000215 2.13255
        2.6978 7.0763 -0.001265 -0.00022 -0.000219 -0.002181
        -0.901853 -9.34643 0 0.010156 -0.000422 -3.09985
        3.95496 6.7923 -0.005035 8e-06 -1e-06 -0
        -8.14702 -4.05581 -1e-05 0 -0 0
        7.32555 -5.48539 4e-06 1.53682 8e-06 -1.4834
        -4.74094 -11.0099 -0.000287 -0.000788 -0.000247 -4.3e-05
        -8.38434 -4.91508 0.006851 -3e-05 -2e-05 -0
        1.5746 6.12876 -9e-06 -2.8e-05 -4.2e-05 0
        -3.86429 -4.51002 -0.011453 0 -0 0
        -6.52727 1.84147 0 0 -0 0
        2.51936 6.64908 0.002745 0 -0 0
        7.54855 -6.4573 0.002665 0 -0 0
        -9.71553 -6.48072 -4.9e-05 -0.000211 -0 -0
        0.37629 6.80542 0.0002 0 -0 0
        1.7087 6.13613 1.46222 -0.043533 0.845955 1.08682
        -8.85188 -1.64045 0.004534 -0.009916 -9.8e-05 -0.00032
        -8.21245 -2.85045 0.007022 0 -0 0
        2.85176 -9.89669 -0.003518 -6e-06 -0.000301 -0.000246
        -3.33041 -4.99334 -0.00654 0.000318 6.6e-05 -9.4e-05
        -1.52755 -8.55652 -0.006702 5.6e-05 0 4e-06
        2.09425 -9.93148 -0.000534 -1.5e-05 -1.9e-05 -4e-06
        7.91003 -8.44886 0.000379 0 -0 0
        2.09307 -8.47055 -0.002354 -8.8e-05 -0 -0
        9.06364 -7.65364 -1e-05 0.018276 -0.001567 1.65632
        -9.67007 5.05675 0 -0.000122 -0 -1.5e-05
        -4.73399 3.43262 0.017586 -0.001833 -3e-06 9.2e-05
        -0.761038 -11.2592 -0.001115 -0.000193 4e-06 4e-06
        -3.38955 -10.4164 0.00607 -4.7e-05 -0 0
        -2.16095 -7.69065 0 -0.010164 0 0.005302
        -9.79011 -5.07137 -0.051675 7e-06 -0.000384 -0.000258
        1.90983 1.53825 -0.122877 -0.064249 -0.081264 -1.55585
        1.86738 7.95043 1e-06 -0.010164 -0 2e-05
        8.57867 -3.23961 1e-06 -0.010179 1e-06 -0.000107
        -9.37614 2.41339 0.069097 -3.9e-05 0.010164 1.56889
        6.25741 -6.52731 0.000192 -6.1e-05 -0 1e-06
        1.03293 -6.15967 0 -0.000397 -0 -0
        1.52271 -4.88382 0.07529 -1.7e-05 -2e-06 4e-06
        -2.93846 -4.62036 -0.000103 -0.009928 -0.004099 0.166283
        -0.410863 -3.60986 -0.0002 -0.000169 -0 -0
        1.37241 -6.58196 -0.003157 1.2e-05 -0 -3.09576
        2.71953 -2.42472 -0.000762 -0 -0 -1.67818
        1.0211 -1.20117 0.010164 0 -0 0
        -1.50151 0.238271 0.001323 -5.5e-05 -3e-06 1.2e-05
        -5.02105 4.05655 -0.000521 -0.0002 -0.000115 -1e-06
        -7.2126 2.25614 0.044753 -0.065995 -6e-06 0.831734
        -6.63535 -2.58851 -0 -4e-05 -4.5e-05 -2.1e-05
        -7.48217 -0.964415 0.105137 -3e-06 -0 1e-05
        -7.33201 -1.87626 0.026959 8.2e-05 -1.4e-05 2e-06
        -5.2076 -2.28126 -0.002646 -6.4e-05 -0 -0
        7.5 -4.5 -0.000759 0 -0 0
        -1.42411 -6.05904 -0.002054 -0.000148 -2e-06 0
        -0.345437 -5.71695 -0.002256 -0.00012 -0 -0
        3.39622 -7.44361 0.000101 -0.000156 8.5e-05 0.000123
        4.60376 -5.10524 -0.000305 0 -0 0
        0.21906 -8.60608 -0.000663 -0 -0 -0
        -0.448886 -6.56186 -0.000886 -0.0002 -0 0
        -2.8315 3.61106 -0.001272 0 -0 0
        31.3164 -4.49925 0.007455 0.000496 0.006892 0.001048
        -2.55668 2.34606 -0.000727 -0.000168 -0 0
        -5.85694 -1.38091 -0.001659 -6.6e-05 -0 -0
        -6.19027 1.72692 -0.004132 7.9e-05 -1.3e-05 -6e-06
        -5.65273 0.616336 0.724734 6.6e-05 -1e-06 0
        -5.20293 -0.512028 2e-06 -0.010209 -3.9e-05 -0.001356
        -4.10204 -0.113875 0 0 -0 0
        -4.29503 -0.586864 0.000245 -0 -0 -0
        6.18993 8.61151 0.152639 -0.001313 -6e-06 1e-06
        1.45037 -11.0176 -0.000661 0 -0 0
        8.51466 -1.47892 0 -0 -0 -0
        8.27656 -0.486616 -0.000152 -0 -0 -0
        1.69356 -6.70118 -0.000196 -0.000377 -0 -0
        2.23478 -5.40887 0 -0.000572 -0 -0
        4.82807 -5.67146 0.791989 0.042362 0.058307 -2.80027
        2.6075 -1.97314 -6e-06 -0.010152 -5e-06 -6e-05
        -0.100705 -10.0734 -0.003448 -7.2e-05 -0 -0
        -9.76347 -4.0691 -0.000266 -2e-06 -1e-06 0
        -3.6101 0.491352 -0.004248 0.000224 -0 0
        -5.68182 -0.16471 0.003787 0 -0 0
        -7.54035 7.47372 0 -0.000166 0 0
        -4.27336 7.4508 0.110303 1e-05 4.3e-05 -1.2e-05
        -6.37146 7.46097 -0.000305 0 -0 0
        0.641013 7.79842 -0.000224 -1e-05 -1e-06 -0
        -2.58398 8.0205 0.005782 -0.000212 0 0
"""


ALL_GOOD_POSES = np.array(list(map(float, ALL_GOOD_POSES_STR.split()))).reshape(-1, 6)


POSE_CORRESPONDING_MODEL = """
    IKEA_bed_BEDDINGE
    IKEA_bed_TROMSO
    IKEA_bookcase_BESTA
    IKEA_bookcase_BILLY
    IKEA_bookcase_EXPEDIT
    IKEA_bookcase_HEMNES
    IKEA_bookcase_KILBY
    IKEA_bookcase_LACK
    IKEA_bookcase_LAIVA
    IKEA_chair_BERNHARD
    IKEA_chair_BORJE
    IKEA_chair_EKENAS
    IKEA_chair_EKTORP
    IKEA_chair_FUSION
    IKEA_chair_HENRIKSDAL
    IKEA_chair_HERMAN
    IKEA_chair_INGOLF
    IKEA_chair_IVAR
    IKEA_chair_JOKKMOKK
    IKEA_chair_JULES
    IKEA_chair_KAUSTBY
    IKEA_chair_KLAPPSTA
    IKEA_chair_MARIUS
    IKEA_chair_MARKUS
    IKEA_chair_PATRIK
    IKEA_chair_POANG
    IKEA_chair_PREBEN
    IKEA_chair_REIDAR
    IKEA_chair_SIGURD
    IKEA_chair_SKRUVSTA
    IKEA_chair_SNILLE
    IKEA_chair_SOLSTA_OLARP
    IKEA_chair_STEFAN
    IKEA_chair_TOBIAS
    IKEA_chair_VILMAR
    IKEA_chair_VRETA
    IKEA_desk_BESTA
    IKEA_desk_EXPEDIT
    IKEA_desk_FREDRIK
    IKEA_desk_GALANT
    IKEA_desk_HEMNES
    IKEA_desk_LAIVA
    IKEA_desk_LEKSVIK
    IKEA_desk_LIATORP
    IKEA_desk_MALM
    IKEA_desk_MICKE
    IKEA_desk_VALLVIK
    IKEA_desk_VIKA
    IKEA_desk_VITTSJO
    IKEA_sofa_EKTORP
    IKEA_sofa_KARLSTAD
    IKEA_sofa_KIVIK
    IKEA_sofa_KLAPPSTA
    IKEA_sofa_KLIPPAN
    IKEA_sofa_LYCKSELE
    IKEA_sofa_MANSTAD
    IKEA_sofa_SATER
    IKEA_sofa_SKOGABY
    IKEA_sofa_SOLSTA
    IKEA_sofa_TIDAFORS
    IKEA_table_BJORKUDDEN
    IKEA_table_BJURSTA
    IKEA_table_BOKSEL
    IKEA_table_DOCKSTA
    IKEA_table_FUSION
    IKEA_table_GRANAS
    IKEA_table_HEMNES
    IKEA_table_INGATORP
    IKEA_table_INGO
    IKEA_table_ISALA
    IKEA_table_JOKKMOKK
    IKEA_table_KLINGSBO
    IKEA_table_KLUBBO
    IKEA_table_LACK
    IKEA_table_LIATORP
    IKEA_table_LINDVED
    IKEA_table_MUDDUS
    IKEA_table_NESNA
    IKEA_table_NORBO
    IKEA_table_NORDEN
    IKEA_table_NORDLI
    IKEA_table_NYVOLL
    IKEA_table_ODDA
    IKEA_table_RAST
    IKEA_table_SALMI
    IKEA_table_TOFTERYD
    IKEA_table_TORSBY
    IKEA_table_UTBY
    IKEA_table_VEJMON
    IKEA_table_VITTSJO
    IKEA_wardrobe_ANEBODA
    IKEA_wardrobe_DOMBAS
    IKEA_wardrobe_HEMNES
    IKEA_wardrobe_ODDA
    IKEA_wardrobe_PAX
""".split()

ALL_IKEA_MODELS = """
            IKEA_bed_TROMSO
            IKEA_bookcase_BESTA
            IKEA_bookcase_BILLY
            IKEA_bookcase_EXPEDIT
            IKEA_bookcase_HEMNES
            IKEA_bookcase_KILBY
            IKEA_bookcase_LACK
            IKEA_bookcase_LAIVA
            IKEA_chair_BERNHARD
            IKEA_chair_BORJE
            IKEA_chair_EKENAS
            IKEA_chair_EKTORP
            IKEA_chair_FUSION
            IKEA_chair_HENRIKSDAL
            IKEA_chair_HERMAN
            IKEA_chair_INGOLF
            IKEA_chair_IVAR
            IKEA_chair_JOKKMOKK
            IKEA_chair_JULES
            IKEA_chair_KAUSTBY
            IKEA_chair_KLAPPSTA
            IKEA_chair_MARIUS
            IKEA_chair_MARKUS
            IKEA_chair_NILS
            IKEA_chair_PATRIK
            IKEA_chair_POANG
            IKEA_chair_PREBEN
            IKEA_chair_REIDAR
            IKEA_chair_SIGURD
            IKEA_chair_SKRUVSTA
            IKEA_chair_SNILLE
            IKEA_chair_SOLSTA_OLARP
            IKEA_chair_STEFAN
            IKEA_chair_TOBIAS
            IKEA_chair_VILMAR
            IKEA_chair_VRETA
            IKEA_desk_BESTA
            IKEA_desk_EXPEDIT
            IKEA_desk_FREDRIK
            IKEA_desk_GALANT
            IKEA_desk_HEMNES
            IKEA_desk_LAIVA
            IKEA_desk_LEKSVIK
            IKEA_desk_LIATORP
            IKEA_desk_MALM
            IKEA_desk_MICKE
            IKEA_desk_VALLVIK
            IKEA_desk_VIKA
            IKEA_desk_VITTSJO
            IKEA_sofa_EKTORP
            IKEA_sofa_KARLSTAD
            IKEA_sofa_KIVIK
            IKEA_sofa_KLAPPSTA
            IKEA_sofa_KLIPPAN
            IKEA_sofa_LYCKSELE
            IKEA_sofa_MANSTAD
            IKEA_sofa_SATER
            IKEA_sofa_SKOGABY
            IKEA_sofa_SOLSTA
            IKEA_sofa_TIDAFORS
            IKEA_sofa_VRETA
            IKEA_table_BJORKUDDEN
            IKEA_table_BJURSTA
            IKEA_table_BOKSEL
            IKEA_table_DOCKSTA
            IKEA_table_FUSION
            IKEA_table_GRANAS
            IKEA_table_HEMNES
            IKEA_table_INGATORP
            IKEA_table_INGO
            IKEA_table_ISALA
            IKEA_table_JOKKMOKK
            IKEA_table_KLINGSBO
            IKEA_table_KLUBBO
            IKEA_table_LACK
            IKEA_table_LIATORP
            IKEA_table_LINDVED
            IKEA_table_MUDDUS
            IKEA_table_NESNA
            IKEA_table_NORBO
            IKEA_table_NORDEN
            IKEA_table_NORDLI
            IKEA_table_NYVOLL
            IKEA_table_ODDA
            IKEA_table_RAST
            IKEA_table_SALMI
            IKEA_table_TOFTERYD
            IKEA_table_TORSBY
            IKEA_table_UTBY
            IKEA_table_VEJMON
            IKEA_table_VITTSJO
            IKEA_wardrobe_ANEBODA
            IKEA_wardrobe_DOMBAS
            IKEA_wardrobe_HEMNES
            IKEA_wardrobe_ODDA
            IKEA_wardrobe_PAX""".split()


def parse_ikea_name(ikea_name, sep="_"):
    _, type_, modelname = ikea_name.split(sep)
    return type_, modelname


available_locations = [
      [-5.06466,-3.30098,1.42,0,-0,0],
      [01.07000,-4.51000,0,0,-0,0],
      [-8.21,6.35,0,0,-0,0],
      [-1.37800,7.51000,0,0,-0,0],
      [-0.907958,-5.34608,0,0,-0,0],
      [-1.33100,-4.556118,0,0,-0,0],
      [3.359000,-3.69000,0,0,-0,-1.544],
      [-9.07891,-2.6341,0,0,-0,0],
      [7.23227,3.2000,0,0,-0,-1.54],
      [7.62048,0.974333,0,0,-0,-1.54],
      [-5.59884,-7.73979,0,0,-0,0],
      [-5.42816,-8.94109,0,0,-0,0],
      [9.94622,-1.61582,0,0,-0,0],
      [-4.234,6.950770,0,0,-0,0]]


def render_jinja_template(jinja_file, dst_file, var_dict):
    template = Template(open(jinja_file).read())
    template.stream(**var_dict).dump(dst_file)


ModelJinja = namedtuple("ModelJinja", ["name", "pose"])


def pose_grid(N, empty_radius, sep):
    side = np.ceil(np.sqrt(N))
    full_grid = np.mgrid[-side/2:side, -side/2:side] * sep
    return full_grid.T.reshape(-1, 2)


def create_gazebo_file(template="world_template.sdf.jinja",
                       dest_file="world_template_generated.sdf",
                       ikea_models=POSE_CORRESPONDING_MODEL,
                       poses=ALL_GOOD_POSES,
                       p=0.6):
    models = []
    #poses = pose_grid(len(ALL_IKEA_MODELS), empty_radius=3, sep=3)
    for i, (ikea_name, pose) in enumerate(zip(ikea_models, poses)):
        pose[3:5] = 0
        if np.random.rand() > p:
            model = ModelJinja(name=ikea_name,
                            pose=tuple(pose.tolist()))
            models.append(model)

    render_jinja_template(relpath(template), relpath(dest_file), dict(models=models))
    return relpath(dest_file)


def create_random_worlds(dest_file_fmt="world%02d.sdf"):
    for i in range(100):
        create_gazebo_file(dest_file=dest_file_fmt % i)


def create_shuffled_worlds(dest_file_fmt="world_shuffled_%02d.sdf"):
    for i in range(100):
        models_copy = POSE_CORRESPONDING_MODEL.copy()
        random.shuffle(models_copy)
        create_gazebo_file(dest_file=dest_file_fmt % i,
                           ikea_models=models_copy,
                           poses=ALL_GOOD_POSES)


if __name__ == '__main__':
    #create_random_worlds()
    create_shuffled_worlds()
