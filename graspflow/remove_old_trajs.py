from pathlib import Path
import arrow
import os

filesPath = "../experiments" # MODIFY TO YOUR FOLDER
criticalTime = arrow.now().shift(hours=+8).shift(days=-1) # 16 hours before

print(criticalTime)

for experiment_num in range(701,725):
    for item in Path(f'{filesPath}/generated_grasps_experiment{experiment_num}/planned_trajectories/').glob('*/*'):
        if item.is_file():
            # print (str(item.absolute()))
            itemTime = arrow.get(item.stat().st_mtime)
            if itemTime < criticalTime:
                # os.remove(item)
                pass

print('Done')
