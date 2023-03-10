from os import link
import numpy as np
import torch
import torch.nn as nn
import time

from robot_model import CombinedRobotModel2


E_classifier = CombinedRobotModel2()


qs = np.array([[ 0.76147866,  0.56797338,  0.29964933, -0.08812968], [ 0.73960179,  0.60007006,  0.2843841,  -0.10968477]])
ts = np.array([[0.67488712, 0.34649354, 0.69040132], [0.6590572,  0.33425328, 0.58144983]])

qs = np.repeat(qs, repeats=100).reshape([-1,4])
ts = np.repeat(ts, repeats=100).reshape([-1,3])

print(qs.shape, ts.shape)

q = torch.FloatTensor(qs)
t = torch.FloatTensor(ts)

start_time = time.time()
res = E_classifier.forward(t,q)

print(f'End_time: {time.time()-start_time}')
print(res.shape)