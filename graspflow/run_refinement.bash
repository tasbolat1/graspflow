#!/bin/bash

# sampler="gpd" # graspnet
# method="metropolis" # GraspFlow, graspnet
# max_iterations=50
# grasp_space="Euler"
# device=1

sampler="$1" 
method="$2"
max_iterations="$3"
grasp_space="$4"
device="$5"
grasp_folder="$6"
include_robot=$7

# # SO3 and Euler
# parser.add_argument("--noise_e", type=float, help="Noise factor for GraspFlow.", default=0.000001)
# parser.add_argument("--noise_t", type=float, help="Noise factor for DFflow.", default=0.000001)
# parser.add_argument("--eta_t", type=float, help="Refiement rate for DFflow.", default=0.000025)
# parser.add_argument("--eta_e", type=float, help="Refiement rate for DFflow.", default=0.000025)

# # Theta
# parser.add_argument("--eta_theta_s", type=float, help="Refinement rate for S classifier for Theta grasp space.", default=0.000025)
# parser.add_argument("--eta_theta_e", type=float, help="Refiement rate for E classifier for Theta grasp space.", default=0.000025)
# parser.add_argument("--noise_theta", type=float, help="Noise factor for DFflow with Theta grasp space.", default=0.000001)

# other parameters

# noise_e=0.000001
# noise_t=0.000001
# eta_t=0.000025
# eta_e=0.000025
# eta_theta_s=0.000025
# eta_theta_e=0.000025
# noise_theta=0.000001

noise_e=$8
noise_t=$9
eta_t=${10}
eta_e=${11}
eta_theta_s=${12}
eta_theta_e=${13}
noise_theta=${14}


refine() {
    python refine_isaac_grasps.py  --cat $1 --idx $2 --max_iterations ${max_iterations} --sampler ${sampler} --method ${method} --device ${device} --grasp_space ${grasp_space} --grasp_folder ${grasp_folder} --include_robot ${include_robot} --noise_e ${noise_e} --noise_t ${noise_t} --eta_e ${eta_e} --eta_t ${eta_t} --noise_theta ${noise_theta} --eta_theta_e ${eta_theta_e} --eta_theta_s ${eta_theta_s}
}


cat="mug"
for idx in 2
    do
     refine ${cat} ${idx}
    done

cat="bowl"
for idx in 16
    do
     refine ${cat} ${idx}
    done

cat="spatula" #
for idx in 1
    do
     refine ${cat} ${idx} 
    done


# cat="box"
# for idx in 14 17
#     do
#     refine ${cat} ${idx}
#     done



# cat="mug"
# for idx in 2 8 14
#     do
#      refine ${cat} ${idx}
#     done


# cat="bowl"
# for idx in 1 16
#     do
#      refine ${cat} ${idx}
#     done


# cat="bottle"
# for idx in 3 12 19
#     do
#      refine ${cat} ${idx} 
#     done


# cat="cylinder"
# for idx in 2 11
#     do
#      refine ${cat} ${idx} 
#     done


# cat="pan"
# for idx in 3 6
#     do
#      refine ${cat} ${idx} 
#     done

# cat="scissor"
# for idx in 4 7
#     do
#      refine ${cat} ${idx} 
#     done

# cat="fork"
# for idx in 1 11
#     do
#      refine ${cat} ${idx} 
#     done


# cat="hammer" # 2 is missing
# for idx in 15
#     do
#      refine ${cat} ${idx} 
#     done


# cat="spatula" #
# for idx in 1 14
#     do
#      refine ${cat} ${idx} 
#     done

