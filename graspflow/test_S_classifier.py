import torch
from networks.models import GraspEvaluator, CollisionDistanceEvaluator
from networks import utils
from networks import quaternion

class GraspEvaluatorCombined(nn.Module):
    def __init__(self, dist_threshold=0.1, dist_coeff=10000, approximate=False, reverse=False):
        super(GraspEvaluatorCombined, self).__init__()

        self.grasp_evalutor = GraspEvaluator()
        self.collision_evaluator = CollisionDistanceEvaluator(dist_coeff=dist_threshold, dist_coeff=dist_coeff, approximate=approximate, reverse=True)