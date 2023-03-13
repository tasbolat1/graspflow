from copy import copy
from fcntl import F_SETFL
import shutil, os
import glob
fs = glob.glob(f"/home/tasbolat/some_python_examples/refinement_experiments/grasp_network/data/grasps_tolerance/*/*_isaac/main_grasps.npz")
copy_dir = '/home/tasbolat/some_python_examples/GRASP/grasp_network/data/grasps_tight'
for f in fs:
    l = f.split('/')[-3:]
    print(f'{copy_dir}/{l[0]}/{l[1]}/{l[2]}')
    shutil.copyfile(f, f'{copy_dir}/{l[0]}/{l[1]}/{l[2]}')