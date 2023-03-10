import torch

import numpy as np
import torch
import math
import json

from theseus.geometry import SO3


class SO3_R3():
    def __init__(self, q=None, t=None):
        self.rot = SO3()
        self.rot.update_from_unit_quaternion(q)
        self.w = self.R.log_map()
        if t is None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H