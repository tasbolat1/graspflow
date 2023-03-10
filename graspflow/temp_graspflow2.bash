#!/bin/bash

device=2
max_iterations=30
batch_size=64

run_graspflow() {
    python refine_isaac_grasps6.py --cat pan --idx 12 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bottle --idx 14 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 10 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat fork --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat mug --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat pan --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
    python refine_isaac_grasps6.py --cat scissor --idx 7 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type shelf006 --batch_size $batch_size
}

run_graspflow 226
run_graspflow 227
run_graspflow 228
run_graspflow 229
run_graspflow 230

run_graspflow 231
run_graspflow 232
run_graspflow 233
run_graspflow 234
run_graspflow 235
