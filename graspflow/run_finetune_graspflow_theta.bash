#!/bin/bash

device=0
grasp_space="Theta"
# eta_theta_s=0.00001
# eta_theta_e=0.00001
# noise_theta=0.000001

robot_threshold=0.0001
robot_coeff=1000

for max_iterations in 10 15 20 25 30 40 50 60 70
do
for include_robot in 1
    do
    for eta_theta_s in 0.0001 0.00025 0.0005 0.00075 0.001
        do
        for eta_theta_e in 0.0 0.00001 0.0001 0.0005  0.001
        do
        for noise_theta in 0.0
        do
        # python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        done
        done  
    done
    done
done

eta_theta_e=0
for max_iterations in 10 15 20 25 30 40 50 60 70
do
for include_robot in 0
    do
    for eta_theta_s in 0.00001 0.0001 0.00025 0.0005 0.00075 0.001
        do
        for noise_theta in 0.0
        do
        # python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --robot_coeff ${robot_coeff} --robot_threshold ${robot_threshold} --include_robot ${include_robot} --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_theta_s ${eta_theta_s} --eta_theta_e ${eta_theta_e} --noise_theta ${noise_theta} --max_iterations ${max_iterations} --device ${device}
        done
    done
    done
done