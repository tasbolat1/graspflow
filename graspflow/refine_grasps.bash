cat=1
#### POLARIS ONLY RUNS DINER ####
declare -a obj=('pan 12' 'spatula 14' 'bottle 0' 'bowl 8' 'fork 6')
declare -a classifier=('SC')
num_samples=10
scene='diner001'
lower=1051
upper=1063
mid=1060
for ((k = $lower ; k <= $upper ; k++ ));
    do
    for i in "${obj[@]}"
        do
            arrIN=(${i// / })
            cat=${arrIN[0]} 
            idx=${arrIN[1]} 
            for j in "${classifier[@]}"
                do  
                    if (($k >= $mid)) ; then  
                        echo "Refining on experiment" $k
                        python refine_isaac_grasps6.py --cat $cat --idx $idx\
                        --grasp_folder ../experiments/generated_grasps_experiment$k \
                        --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 30 \
                        --method GraspOptES --classifier $j --experiment_type $scene \
                        --cfg configs/graspopt_isaac_params.yaml --batch_size 15
                    else
                        python refine_isaac_grasps6.py --cat $cat --idx $idx\
                        --grasp_folder ../experiments/generated_grasps_experiment$k \
                        --sampler graspnet --grasp_space SO3 --device 0 --max_iterations 0 \
                        --method GraspOptES --classifier $j --experiment_type $scene \
                        --cfg configs/graspopt_isaac_params.yaml
                    fi
                done
        done
    done