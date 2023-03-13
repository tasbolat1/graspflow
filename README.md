# GraspFlow

## Introduction
In this work, we present GraspFlow, a refinement approach for generating context-specific grasps. We formulate the problem of grasp synthesis as a sampling problem: we seek to sample from a context-conditioned probability distribution of successful grasps. However, this target distribution is unknown. As a solution, we devise a discriminator gradient-flow method to evolve grasps obtained from a simpler distribution in a manner that mimics sampling from the desired target distribution. Unlike existing approaches, GraspFlow is modular, allowing grasps that satisfy multiple criteria to be obtained simply by incorporating the relevant discriminators. It is also simple to implement, requiring minimal code given existing auto-differentiation libraries and suitable discriminators.Experiments show that GraspFlow generates stable and executable grasps on a real-world Panda robot for a diverse range of objects.
In particular, in 60 trials on 20 different household objects, the first attempted grasp was successful 94% of the time, and 100% grasp success was achieved by the second grasp. Moreover, incorporating a functional discriminator for robot-
human handover improved the functional aspect of the grasp by up to 33%.

<!-- ![mainfigure](figs/mainfig-1.png =400x) -->

<!-- <img src="figs/mainfig-1.png" alt="main" width="600" height="300"> -->
<img align="center" alt="GraspFlow" src="figs/mainfig-1.png" width="710" height="435" />

This project has a lot of submodules. We recommend to use forked versions of submodules given in this repository. 

## Section 1: Stability Classifier: Dataset and Training

The core of the dataset generation lies in the GraspSampler library. The library can be downloaded here: [GraspSampler](https://github.com/patrickeala/GraspSampler).

```
TODO Pat
```

In order to generate specific grasps per category, please use the following codes:

```
TODO Zhang
```

Or, you can just download the dataset from [google drive](https://drive.google.com/drive/folders/1inXxyXslszR9PW44TQY7v08xZZUiMExS?usp=sharing). After downloading, unzip under graspflow folder. The directory tree must look like as following:

```
 - graspflow
    - graspflow
        - data
            - grasps_lower
            - grasps_tight_lower
            - pcs
```

To train Stability Classifier, please download the dataset, please run the following command:

```
python train_evaluator.py --data_dir_pcs data/pcs --batch_size 512 --lr 0.0001 --device_ids 0 1
```

*Note*: We trained the classifier using 2 NVIDIA RTX3090. Lr gradually decreased overtime. Please have a look tensorboard output for the learnign curve.

## Section 2: GraspNet from NVidia

We use Pytorch version [2] of the Graspnet [1]. 
### Prerequisits:
1. Pointnet2_PyTorch (given as a submodule in this repo) - PointNet++ library for GraspNet's backbone.
2. franka_analytical_ik [3] - solves analytical IK for Panda Robot.

### Installation
We mainly follow same installation as in [2]. However, we also extended it to add additional filtering capabilites. Please install IK submodule and copy generated library to pytorch_6dof-graspnet module. Details are given in [this link](https://github.com/tasbolat1/franka_analytical_ik.git).

*Note:* Generally any grasp sampler can be used. In our paper, we have also tested on [GPD sampler](https://github.com/tasbolat1/gpd.git).


## Section 3: Grasp Refinement via GraspFlow

### Prerequisites
1. franka_analytical_ik [3] - solves analytical IK for Panda Robot.
2. differentiable-robot-model [4] -  differentiable robot model used for E-classifier to calculate FK of the robot.

### Usage
To refine grasps, go to graspflow folder:

```
cd graspflow/graspflow
```

and run:
```
python refine_isaac_samples.py --sampler graspnet --eta_trans 0.00001 --eta_rots 0.00000001 --cat <cat> --idx <idx> --max_iterations 50 --device 0 --f KL --method <M_TYPE> --grasp_folder ../experiments/test
```
For list of parameters for the above function, please check settings.txt file.

## BibTeX

To cite this work, please use:

```
TODO
```

## Reference List
[1]. Mousavian, Arsalan, Clemens Eppner, and Dieter Fox. "6-dof graspnet: Variational grasp generation for object manipulation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019. [Original GitHub repo](https://github.com/NVlabs/6dof-graspnet.git)

[2]. Jens Lundell. "6-DOF GraspNet Pytorch". 2020. [Original GitHub repo](https://github.com/jsll/pytorch_6dof-graspnet.git)

[3]. He, Yanhao, and Steven Liu. "Analytical inverse kinematics for franka emika panda–a geometrical solver for 7-dof manipulators with unconventional design." 2021 9th International Conference on Control, Mechatronics and Automation (ICCMA). IEEE, 2021. [Original GitHub repo](https://github.com/ffall007/franka_analytical_ik.git)

[4]. Sutanto, Giovanni, et al. "Encoding physical constraints in differentiable newton-euler algorithm." Learning for Dynamics and Control. PMLR, 2020.
