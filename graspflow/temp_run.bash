#!/bin/bash

python refine_isaac_grasps3.py --cat box --idx 14 --sampler gpd --device 1 --max_iterations 100 --method GraspFlow --grasp_space SO3 --classifier ST --eta_t 0.005 --eta_e 0.005 --eta_table_t 0.005 --eta_table_e 0.005 --grasp_folder ../experiments/generated_grasps_experiment31 
python refine_isaac_grasps3.py --cat spatula --idx 1 --sampler gpd --device 1 --max_iterations 100 --method GraspFlow --grasp_space SO3 --classifier ST --eta_t 0.005 --eta_e 0.005 --eta_table_t 0.005 --eta_table_e 0.005 --grasp_folder ../experiments/generated_grasps_experiment31 
python refine_isaac_grasps3.py --cat mug --idx 2 --sampler gpd --device 1 --max_iterations 100 --method GraspFlow --grasp_space SO3 --classifier ST --eta_t 0.005 --eta_e 0.005 --eta_table_t 0.005 --eta_table_e 0.005 --grasp_folder ../experiments/generated_grasps_experiment31 
python refine_isaac_grasps3.py --cat cylinder --idx 11 --sampler gpd --device 1 --max_iterations 100 --method GraspFlow --grasp_space SO3 --classifier ST --eta_t 0.005 --eta_e 0.005 --eta_table_t 0.005 --eta_table_e 0.005 --grasp_folder ../experiments/generated_grasps_experiment31 
