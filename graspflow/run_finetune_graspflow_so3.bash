#!/bin/bash

device=2
grasp_space="SO3"
noise_t=0.0
noise_e=0.0
#max_iterations=50


# for max_iterations in 10 15 20 25 30 40 50 60 70
#     do
#     for eta_t in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001
#         do
#         for eta_e in 0.00001 0.00005 0.0001 0.0005 0.001
#         do
#         python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#         python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#         python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#         done  
#     done
# done


for max_iterations in 10
    do
    for eta_t in 0.0001
        do
        for eta_e in 0.001 0.0025 0.005
        do
        python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        done  
    done
done

for max_iterations in 30 40 50
    do
    for eta_t in 0.00025
        do
        for eta_e in 0.001 0.0025 0.005
        do
        python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        done  
    done
done

for max_iterations in 60 70
    do
    for eta_t in 0.0001
        do
        for eta_e in 0.001 0.0025 0.005
        do
        python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
        done  
    done
done

# max_iterations=10
# for eta_t in 0.000070
#     do
#     for eta_e in 0.00500
#     do
#     for noise_t in 0.0
#         do
#         for noise_e in 0.0
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done

# max_iterations=20
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done

# max_iterations=30
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done

# max_iterations=40
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done

# max_iterations=50
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done


# max_iterations=60
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#             done
#     done
#     done


# max_iterations=70
# for eta_t in 0.00005 0.000075 0.00010 0.00025
#     do
#     for eta_e in 0.00001 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005
#     do
#     for noise_t in 0.0 0.00001 0.0001 0.001 0.01
#         do
#         for noise_e in 0.0 0.00001 0.0001 0.001 0.01
#             do
#             python refine_isaac_grasps.py --sampler gpd --cat mug --idx 2 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat bowl --idx 16 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             python refine_isaac_grasps.py --sampler gpd --cat spatula --idx 1 --method GraspFlow --grasp_space ${grasp_space} --eta_e ${eta_e} --eta_t ${eta_t} --noise_e ${noise_e} --noise_t ${noise_t} --max_iterations ${max_iterations} --device ${device}
#             done
#         done
#     done
#     done