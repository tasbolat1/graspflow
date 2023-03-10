cat=1
declare -a obj=('mug 8')
declare -a classifier=('SECN')
for j in "${classifier[@]}"
    do
    for i in "${obj[@]}"
        do
            arrIN=(${i// / })
            cat=${arrIN[0]} 
            idx=${arrIN[1]} 
            # python generate_data_from_isaac_pcs.py --batch_size 10 \
            # --cat $cat --idx $idx --num_grasp_samples 30 --refinement_method gradient \
            # --refine_steps 10 --save_dir ../experiments/generated_grasps_experiment86 \
            # --experiment_type diner001

            python refine_isaac_grasps6.py --cat $cat --idx $idx\
                    --grasp_folder ../experiments/generated_grasps_experiment86 \
                    --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 30 \
                    --method GraspOptES --classifier $j --experiment_type diner001 \
                    --cfg configs/graspflow_isaac_params_N.yaml --batch_size 15

            
            cd ~/
            source /opt/ros/noetic/setup.bash
            source ~/ws_moveit/devel/setup.bash

            rosrun graspflow_plan test_robot_complex_modified.py \
            $cat $idx graspnet GraspOptES SECN SO3 \
            /home/tasbolat/some_python_examples/graspflow_models/experiments/generated_grasps_experiment86 1
        done    
done


