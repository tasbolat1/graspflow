import numpy as np
import trimesh

def gripper_bd(quality=None):
    gripper_line_points_main_part = np.array([
        [0.0401874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0401874312758446, -0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0401874312758446, 0.0000599553131906, 0.1055731475353241],
    ])


    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    

    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
    else:
        color = None
    small_gripper_main_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0,1,2,3,4], color=color)],
                                                vertices = gripper_line_points_main_part)
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                                vertices = gripper_line_points_handle_part)
    small_gripper = trimesh.path.util.concatenate([small_gripper_main_part,
                                    small_gripper_handle_part])
    return small_gripper

