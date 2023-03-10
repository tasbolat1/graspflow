cat=1
#declare -a obj=('pan 12' 'spatula 14' 'bottle 0' 'bowl 8' 'fork 6')
declare -a obj=('bottle 14' 'bowl 8' 'bowl 10' 'pan 6' 'pan 12' 'fork 6' 'scissor 7')
declare -a classifier=('SECN')
num_samples=10
scene='shelf008'
export CUDA_VISIBLE_DEVICES=2
for j in "${classifier[@]}"
    do
        for i in "${obj[@]}"
            do
                arrIN=(${i// / })
                cat=${arrIN[0]} 
                idx=${arrIN[1]} 
                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment611 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15

                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment612 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15
                
                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment613 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15

                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment614 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15
                
                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment615 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15

                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment618 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 30 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15

                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment621 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 30 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15

                python refine_isaac_grasps6.py --cat $cat --idx $idx\
                 --grasp_folder ../experiments/generated_grasps_experiment624 \
                 --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 30 \
                 --method GraspOptES --classifier $j --experiment_type $scene \
                 --cfg configs/graspopt_isaac_params.yaml --batch_size 15
            done    
    done



