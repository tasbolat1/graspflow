#!/bin/bash

device=1
max_iterations=0
batch_size=64
experiment_type='shelf008'

run_graspflow() {
    python refine_isaac_grasps6.py --cat pan --idx 12 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bottle --idx 14 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 10 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat fork --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    # python refine_isaac_grasps6.py --cat mug --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat pan --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat scissor --idx 7 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
}


# run_graspflow 301
# run_graspflow 302
# run_graspflow 303
# run_graspflow 304
# run_graspflow 305

run_graspflow 306
run_graspflow 307
run_graspflow 308
run_graspflow 309
run_graspflow 310

run_graspflow 311
run_graspflow 312
run_graspflow 313
run_graspflow 314
run_graspflow 315

device=1
max_iterations=29
batch_size=64
experiment_type='shelf008'

run_graspflow() {
    python refine_isaac_grasps6.py --cat pan --idx 12 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bottle --idx 14 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat bowl --idx 10 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat fork --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    # python refine_isaac_grasps6.py --cat mug --idx 8 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat pan --idx 6 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
    python refine_isaac_grasps6.py --cat scissor --idx 7 --grasp_folder ../experiments/generated_grasps_experiment$1 --sampler graspnet --grasp_space SO3 --device $device --max_iterations $max_iterations --method GraspOptES --classifier SEC --experiment_type $experiment_type --batch_size $batch_size
}

# run_graspflow 316
# run_graspflow 321
# run_graspflow 326

run_graspflow 317
run_graspflow 318

run_graspflow 322
run_graspflow 323

run_graspflow 327
run_graspflow 328


# run_graspflow 101
# run_graspflow 102
# run_graspflow 103
# run_graspflow 104
# run_graspflow 105
# run_graspflow 106
# run_graspflow 107
# run_graspflow 108
# run_graspflow 109
# run_graspflow 110
# run_graspflow 111
# run_graspflow 112
# run_graspflow 113
# run_graspflow 114
# run_graspflow 115
# run_graspflow 116
# run_graspflow 117
# run_graspflow 118
# run_graspflow 119
# run_graspflow 120
# run_graspflow 121
# run_graspflow 122
# run_graspflow 123
# run_graspflow 124
# run_graspflow 125

# run_graspflow 126
# run_graspflow 127
# run_graspflow 128
# run_graspflow 129
# run_graspflow 130

# run_graspflow 201
# run_graspflow 202
# run_graspflow 203
# run_graspflow 204
# run_graspflow 205
# run_graspflow 206
# run_graspflow 207
# run_graspflow 208
# run_graspflow 209
# run_graspflow 210


# run_graspflow 211
# run_graspflow 212
# run_graspflow 213
# run_graspflow 214
# run_graspflow 215

# run_graspflow 216
# run_graspflow 217
# run_graspflow 218
# run_graspflow 219

# run_graspflow 220
# run_graspflow 221
# run_graspflow 222
# run_graspflow 223
# run_graspflow 224
# run_graspflow 225