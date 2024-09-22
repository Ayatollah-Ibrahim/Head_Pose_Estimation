import os
import pandas as pd
from image_processing import extract_face_mesh
from pose_estimation import extract_pose_params

def create_dataset(jpeg_folder, mat_folder):
    """Create a dataset from images and corresponding .mat files."""
    dataset = []
    for jpeg_file in os.listdir(jpeg_folder):
        if jpeg_file.endswith(('.jpg', '.jpeg')):
            jpeg_path = os.path.join(jpeg_folder, jpeg_file)
            mat_path = os.path.join(mat_folder, jpeg_file.rsplit('.', 1)[0] + '.mat')

            face_mesh_points = extract_face_mesh(jpeg_path)
            if os.path.exists(mat_path):
                yaw, pitch, roll = extract_pose_params(mat_path)
                dataset.append({
                    'image': jpeg_file,
                    'face_mesh': face_mesh_points,
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll
                })

    return pd.DataFrame(dataset)
