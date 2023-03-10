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

def quality2color(quality):

    if quality is None:
        return None

    if isinstance(quality, float) or isinstance(quality, int):
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        return [_R, _G, _B, 1.0]
    else:
        color = []
        for q in quality:
            _B = q*1.0
            _R = 1.0-_B
            _G = 0
            color.append([_R, _G, _B, 1.0])
        
        color = np.array(color)
        return color


def gripper_vector(g=None, size=0.5, length=1, quality=None):
    '''
    Creates 3D vector for the grasp

    Arguments:
        - g [4,4]: transform of the grasp
        - size: size of the vector
        - length: length of the vector
        - quality: determines color of the vector (from blue=1 to red=0)
    Returns:
        - vector3D: trimesh that can be directly used as a geometry
    '''

    # define sizes
    tail_size=0.01*size
    body_size=0.005*size
    head_size=0.01*size
    body_length=0.1*length

    # define colors
    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
    else:
        color = None

    sphere = trimesh.primitives.Sphere(center=(g[:3,3]), radius=tail_size, color=quality)

    g_cyl = np.copy(g)
    standoff_mat = np.eye(4)
    standoff_mat[2] = body_length/2
    g_cyl[:3,3] = np.matmul(g,standoff_mat)[:3,3]

    cylinder = trimesh.primitives.Cylinder(height=body_length, transform=g_cyl, radius=body_size)
    g_cone = np.copy(g)
    standoff_mat = np.eye(4)
    standoff_mat[2] = body_length
    g_cone[:3,3] = np.matmul(g,standoff_mat)[:3,3]
    cone = trimesh.creation.cone(radius=head_size, height=0.01, transform=g_cone)

    vector3D = trimesh.util.concatenate([sphere, cylinder, cone])
    vector3D.visual.face_colors = color

    return vector3D