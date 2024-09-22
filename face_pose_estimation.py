# main.py

# -*- coding: utf-8 -*-
"""
Face Pose Estimation Using AFLW2000-3D Dataset
This script estimates the pose of a face in images using a dataset
containing facial landmarks and corresponding pose parameters (yaw, pitch, roll).
It utilizes MediaPipe for face mesh extraction and employs various regression models
to predict pose angles based on extracted features.
"""

import os
import random
import math
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Initialize MediaPipe FaceMesh for landmark extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_face_mesh(image_path):
    """Extract face mesh landmarks from an image.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: A dictionary containing the coordinates of selected face landmarks.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    interested_indices = [1, 33, 263, 61, 291, 10, 152, 168]
    landmark_coords = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in interested_indices:
                landmark = face_landmarks.landmark[idx]
                x, y, z = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]), landmark.z
                landmark_coords[idx] = (x, y, z)

    return landmark_coords

def extract_pose_params(mat_file_path):
    """Extract yaw, pitch, and roll from a .mat file.
    
    Args:
        mat_file_path (str): Path to the input .mat file.

    Returns:
        tuple: A tuple containing yaw, pitch, and roll angles.
    """
    mat_data = sio.loadmat(mat_file_path)
    pose_para = mat_data.get('Pose_Para', [])
    if pose_para.size > 0:
        return pose_para[0][:3]  # Return yaw, pitch, roll
    return None, None, None

def create_dataset(jpeg_folder, mat_folder):
    """Create a dataset from images and corresponding .mat files.
    
    Args:
        jpeg_folder (str): Path to the folder containing JPEG images.
        mat_folder (str): Path to the folder containing MAT files.

    Returns:
        DataFrame: A pandas DataFrame containing the dataset with image names, 
                    face mesh landmarks, and pose parameters.
    """
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

def compute_pitch_features(face_mesh):
    """Compute pitch features from face mesh landmarks.
    
    Args:
        face_mesh (dict): Dictionary of face mesh landmarks.

    Returns:
        tuple: Two ratios representing pitch features.
    """
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    d10_1 = euclidean(points[10], points[1])
    d10_152 = euclidean(points[10], points[152])
    d1_152 = euclidean(points[1], points[152])

    ratio1 = d10_1 / d10_152 if d10_152 != 0 else 0
    ratio2 = d1_152 / d10_152 if d10_152 != 0 else 0

    return ratio1, ratio2

def compute_yaw_features(face_mesh):
    """Compute yaw features from face mesh landmarks.
    
    Args:
        face_mesh (dict): Dictionary of face mesh landmarks.

    Returns:
        tuple: Two ratios representing yaw features.
    """
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    d33_168 = euclidean(points[33], points[168])
    d33_263 = euclidean(points[33], points[263])
    d168_263 = euclidean(points[168], points[263])

    ratio1 = d33_168 / d33_263 if d33_263 != 0 else 0
    ratio2 = d168_263 / d33_263 if d33_263 != 0 else 0

    return ratio1, ratio2

def compute_roll_feature(face_mesh):
    """Compute roll feature from face mesh landmarks.
    
    Args:
        face_mesh (dict): Dictionary of face mesh landmarks.

    Returns:
        float: Roll angle in radians.
    """
    points = {index: np.array(coords) for index, coords in face_mesh.items()}
    vector = points[10] - points[1]
    vertical_vector = np.array([0, 1, 0])

    cos_theta = np.dot(vector, vertical_vector) / (np.linalg.norm(vector) * np.linalg.norm(vertical_vector))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle_rad

def evaluate_models(X, y):
    """Evaluate various regression models on the provided features and target.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target values.

    Returns:
        dict: A dictionary containing model performance metrics.
    """
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Elastic Net': ElasticNet(random_state=42),
    }
    results = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results[model_name] = {'MSE': mse, 'R²': r2, 'Mean CV MSE': mean_cv_score}
    return results

def draw_axis(img, yaw, pitch, roll, nose_tip, size=100):
    """Draw axes on the image based on yaw, pitch, and roll angles.
    
    Args:
        img (ndarray): The image on which to draw.
        yaw (float): Yaw angle.
        pitch (float): Pitch angle.
        roll (float): Roll angle.
        nose_tip (tuple): Coordinates of the nose tip.
        size (int): Length of the axes to be drawn.

    Returns:
        ndarray: The image with drawn axes.
    """
    tdx, tdy = nose_tip

    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-Axis
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-Axis
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)  # Z-Axis

    return img

def display_image_with_axes(jpeg_folder, dataset, pitch_model, yaw_model, roll_model):
    """Display a random image with predicted axes based on model predictions.
    
    Args:
        jpeg_folder (str): Path to the folder containing JPEG images.
        dataset (DataFrame): The dataset containing images and pose parameters.
        pitch_model: The trained model for predicting pitch.
        yaw_model: The trained model for predicting yaw.
        roll_model: The trained model for predicting roll.
    """
    random_image = random.choice(dataset['image'].tolist())
    jpeg_path = os.path.join(jpeg_folder, random_image)

    img = cv2.imread(jpeg_path)
    face_mesh_points = extract_face_mesh(jpeg_path)

    nose_tip_index = 1  # Index for the nose tip
    nose_tip = (face_mesh_points[nose_tip_index][0], face_mesh_points[nose_tip_index][1])

    pitch_ratios = compute_pitch_features(face_mesh_points)
    yaw_ratios = compute_yaw_features(face_mesh_points)
    roll_feature = compute_roll_feature(face_mesh_points)

    pitch_input = np.array([[pitch_ratios[0], pitch_ratios[1]]])
    yaw_input = np.array([[yaw_ratios[0], yaw_ratios[1]]])
    roll_input = np.array([[roll_feature]])

    pitch_pred = pitch_model.predict(pitch_input)[0]
    yaw_pred = yaw_model.predict(yaw_input)[0]
    roll_pred = roll_model.predict(roll_input)[0]

    original_values = dataset[dataset['image'] == random_image].iloc[0]
    original_yaw = original_values['yaw']
    original_pitch = original_values['pitch']
    original_roll = original_values['roll']

    img_with_axes = draw_axis(img, yaw_pred, pitch_pred, roll_pred, nose_tip)
    img_rgb = cv2.cvtColor(img_with_axes, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Predicted Yaw: {yaw_pred:.2f}, Original Yaw: {original_yaw:.2f}\n'
              f'Predicted Pitch: {pitch_pred:.2f}, Original Pitch: {original_pitch:.2f}\n'
              f'Predicted Roll: {roll_pred:.2f}, Original Roll: {original_roll:.2f}')
    plt.show()

def main():
    """Main function to run face pose estimation.
    
    This function orchestrates the entire process from loading data, training models,
    and displaying results.
    """
    try:
        jpeg_folder = './data/JPEG'  # Path to your local JPEG folder
        mat_folder = './data/MAT'     # Path to your local MAT folder

        df = create_dataset(jpeg_folder, mat_folder)
        
        if df.empty:
            print("No valid data found.")
            return
        
        df = df[df['face_mesh'].apply(lambda x: len(x) > 0)]
        
        # Compute pitch, yaw, and roll features
        df[['pitch_ratio1', 'pitch_ratio2']] = df['face_mesh'].apply(lambda x: pd.Series(compute_pitch_features(x)))
        df[['yaw_ratio1', 'yaw_ratio2']] = df['face_mesh'].apply(lambda x: pd.Series(compute_yaw_features(x)))
        df['roll_angle'] = df['face_mesh'].apply(compute_roll_feature)
        
        # Prepare feature sets and target variables
        X_pitch = df[['pitch_ratio1', 'pitch_ratio2']]
        y_pitch = df['pitch']
        X_yaw = df[['yaw_ratio1', 'yaw_ratio2']]
        y_yaw = df['yaw']
        X_roll = df[['roll_angle']]
        y_roll = df['roll']
        
        # Train models for pitch, yaw, and roll
        pitch_model = XGBRegressor(random_state=42)
        pitch_model.fit(X_pitch, y_pitch)

        yaw_model = XGBRegressor(random_state=42)
        yaw_model.fit(X_yaw, y_yaw)

        roll_model = XGBRegressor(random_state=42)
        roll_model.fit(X_roll, y_roll)

        # Display an image with predicted axes
        display_image_with_axes(jpeg_folder, df, pitch_model, yaw_model, roll_model)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
