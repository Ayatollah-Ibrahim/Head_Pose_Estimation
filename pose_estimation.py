import scipy.io as sio
import numpy as np
from scipy.spatial.distance import euclidean

def extract_pose_params(mat_file_path):
    """Extract yaw, pitch, and roll from a .mat file."""
    mat_data = sio.loadmat(mat_file_path)
    pose_para = mat_data.get('Pose_Para', [])
    if pose_para.size > 0:
        return pose_para[0][:3]  # Return yaw, pitch, roll
    return None, None, None

def compute_pitch_features(face_mesh):
    """Compute pitch features from face mesh landmarks."""
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    d10_1 = euclidean(points[10], points[1])
    d10_152 = euclidean(points[10], points[152])
    d1_152 = euclidean(points[1], points[152])

    ratio1 = d10_1 / d10_152 if d10_152 != 0 else 0
    ratio2 = d1_152 / d10_152 if d10_152 != 0 else 0

    return ratio1, ratio2

def compute_yaw_features(face_mesh):
    """Compute yaw features from face mesh landmarks."""
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    d33_168 = euclidean(points[33], points[168])
    d33_263 = euclidean(points[33], points[263])
    d168_263 = euclidean(points[168], points[263])

    ratio1 = d33_168 / d33_263 if d33_263 != 0 else 0
    ratio2 = d168_263 / d33_263 if d33_263 != 0 else 0

    return ratio1, ratio2

def compute_roll_feature(face_mesh):
    """Compute roll feature from face mesh landmarks."""
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    vector = points[10] - points[1]
    vertical_vector = np.array([0, 1, 0])

    cos_theta = np.dot(vector, vertical_vector) / (np.linalg.norm(vector) * np.linalg.norm(vertical_vector))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle_rad
