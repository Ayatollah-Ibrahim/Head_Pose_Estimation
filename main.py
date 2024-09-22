import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from dataset import create_dataset
from image_processing import draw_axis, extract_face_mesh
from pose_estimation import compute_pitch_features, compute_yaw_features, compute_roll_feature

def display_image_with_axes(jpeg_folder, dataset, pitch_model, yaw_model, roll_model):
    """Display a random image with predicted axes based on model predictions."""
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
    """Main function to run face pose estimation."""
    try:
        jpeg_folder = './data/JPEG'  # Path to your local JPEG folder
        mat_folder = './data/MAT'     # Path to your local MAT folder

        df = create_dataset(jpeg_folder, mat_folder)
        
        if df.empty:
            print("No valid data found.")
            return
        
        df = df[df['face_mesh'].apply(lambda x: len(x) > 0)]
        
        df[['pitch_ratio1', 'pitch_ratio2']] = df['face_mesh'].apply(lambda x: pd.Series(compute_pitch_features(x)))
        df[['yaw_ratio1', 'yaw_ratio2']] = df['face_mesh'].apply(lambda x: pd.Series(compute_yaw_features(x)))
        df['roll_angle'] = df['face_mesh'].apply(compute_roll_feature)
        
        X_pitch = df[['pitch_ratio1', 'pitch_ratio2']]
        y_pitch = df['pitch']
        X_yaw = df[['yaw_ratio1', 'yaw_ratio2']]
        y_yaw = df['yaw']
        X_roll = df[['roll_angle']]
        y_roll = df['roll']
        
        # Split the data into training and testing sets
        X_train_pitch, X_test_pitch, y_train_pitch, y_test_pitch = train_test_split(X_pitch, y_pitch, test_size=0.2, random_state=42)
        X_train_yaw, X_test_yaw, y_train_yaw, y_test_yaw = train_test_split(X_yaw, y_yaw, test_size=0.2, random_state=42)
        X_train_roll, X_test_roll, y_train_roll, y_test_roll = train_test_split(X_roll, y_roll, test_size=0.2, random_state=42)

        # Train models for pitch, yaw, and roll
        pitch_model = XGBRegressor(random_state=42)
        pitch_model.fit(X_train_pitch, y_train_pitch)

        yaw_model = XGBRegressor(random_state=42)
        yaw_model.fit(X_train_yaw, y_train_yaw)

        roll_model = XGBRegressor(random_state=42)
        roll_model.fit(X_train_roll, y_train_roll)

        # Optionally, evaluate the models on the test set
        pitch_pred = pitch_model.predict(X_test_pitch)
        yaw_pred = yaw_model.predict(X_test_yaw)
        roll_pred = roll_model.predict(X_test_roll)

        display_image_with_axes(jpeg_folder, df, pitch_model, yaw_model, roll_model)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
