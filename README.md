# **Head Pose Estimation**

## **Overview**

This repository contains a Python implementation for estimating the yaw, pitch, and roll of human faces in images using 3D facial landmark detection. The project is based on a subset of the AFLW2000-3D dataset, utilizing 64 images annotated with .mat files, from which the 3D facial landmarks are extracted for head pose estimation.

## **Dataset**

### **AFLW2000-3D Subset**

For this project, a subset of 64 images from the AFLW2000-3D dataset is used, which contains images annotated with 68 facial landmarks in 3D. These annotations provide crucial information for training and evaluating head pose estimation models.

- **Citation**:  
  ```
  Zhu, X., Wang, Z., & Liu, Y. (2016). Face Alignment Across Large Poses: A 3D Solution. In European Conference on Computer Vision (ECCV).
  ```

- **Source**: [AFLW2000-3D on TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/aflw2k3d)

## **Project Structure**

```
.
├── data/
│   ├── JPEG/            # Folder containing 64 images used in the project
│   ├── MAT/             # Folder containing .mat files with 3D landmark annotations
├── .gitignore           # Git ignore file
├── LICENSE              # License file (MIT)
├── README.md            # This README file
├── dataset.py           # Script to load and preprocess dataset
├── image_processing.py  # Script for image processing and feature extraction
├── main.py              # Main script to run the pose estimation pipeline
├── model_evaluation.py  # Script to evaluate the trained model
├── pose_estimation.py   # Script for model training and prediction
├── requirements.txt     # Python package dependencies
```

## **Installation**

To set up this project, follow the steps below:

### 1. **Clone the Repository**

```bash
git clone https://github.com/Ayatollah-Ibrahim/Head_Pose_Estimation.git
cd face-pose-estimation
```

### 2. **Install Required Libraries**

Install the necessary Python libraries by running the following command:

```bash
pip install -r requirements.txt
```

Ensure you are using Python 3.6 or higher.

### 3. **Dataset Setup**

Ensure you have the following folder structure inside the `data` folder:

- `JPEG/`: Contains 64 facial images.
- `MAT/`: Contains .mat files with the 3D annotations corresponding to the images in the `JPEG/` folder.

These annotations will be used by the model to extract facial landmarks.

## **Usage**

### **Running the Code**

1. **Dataset Preparation**: Ensure the images and .mat files are in the correct folders (`JPEG/` and `MAT/`) under the `data/` directory.
2. **Start Pose Estimation**: To run the main pipeline, execute the following command:

   ```bash
   python main.py
   ```

### **Code Breakdown**

- `dataset.py`: 
  - Contains the `create_dataset()` function to load images and their corresponding 3D landmark annotations from the .mat files.
  
- `image_processing.py`: 
  - Handles image pre-processing and extraction of 68 3D landmarks. 
  - Also computes geometric features like distances and angles between key facial points, which are crucial for pose estimation.
    ### **Yaw, Pitch, and Roll Feature Extraction**:
    
    Let’s rewrite the formulas using plain text and symbols that work in markdown files. Here's the updated version:
    
    
    **Facial Landmark Points for Feature Extraction**
    In this project, we utilize specific facial landmarks to accurately estimate the yaw, pitch, and roll angles of human faces. The following landmark points, extracted from the AFLW2000-3D dataset, form the basis of our feature extraction process:
    
    **Landmark Definitions**
    
    1. **Nose Tip (Index 1)**: This point serves as a central reference for facial alignment.
    2. **Forehead (Index 10)**: Located on the forehead, this point helps provide vertical orientation data.
    3. **Chin (Index 152)**: The bottom point of the face, crucial for measuring vertical distances.
    4. **Middle of the Face (Index 168)**: Positioned between the two eyes at the nose level, this point aids in assessing horizontal alignment.
    5. **Left Eye's Outer Corner (Index 263)**: The outer corner of the left eye, used for calculating angles and distances related to head orientation.
    6. **Right Eye's Outer Corner (Index 33)**: The outer corner of the right eye, which assists in the assessment of yaw.
       
    ![canonical_face_model_uv_visualization](https://github.com/user-attachments/assets/dcfacd6b-5a1f-4598-8014-3b8a34e69bb4)

    **Feature Extraction Process**
    
    The geometric relationships among these landmarks allow us to derive meaningful features for head pose estimation:
    
    - **Pitch Calculation**: 
      - We compute the distances between the chin, nose tip, and forehead. Ratios of these distances give us a normalized measure of the vertical alignment of facial features, which is essential for estimating the pitch angle.
    
    - **Yaw Calculation**: 
      - The distances between the corners of the eyes (indices 33 and 263) and the middle of the face (index 168) are calculated. Ratios of these distances quantify the horizontal alignment of facial features, providing the necessary information to estimate the yaw angle.
    
    - **Roll Calculation**: 
      - The direction vector from the chin (index 152) to the nose tip (index 1) is computed and compared with a vertical vector. The angle formed gives us the roll of the head, indicating how much the face is tilted to either side.
    
    **How to Use These Points**
    
    1. **Extraction**: 
       - The landmark coordinates are extracted from the dataset and stored in a dictionary format for easy access.
    
    2. **Feature Functions**: 
       - The functions `compute_pitch_features`, `compute_yaw_features`, and `compute_roll_feature` are implemented to calculate the respective ratios and angles using the defined landmarks.
    
    3. **Integration**: 
       - These extracted features are then used as input for machine learning models (e.g., Random Forest, XGBoost) to predict the head pose accurately.
    
    By leveraging these specific facial landmarks, our feature extraction approach provides a robust framework for estimating head poses, facilitating applications in facial recognition, driver monitoring, and augmented reality.


- `pose_estimation.py`: 
  - This script contains the models used for predicting yaw, pitch, and roll. The landmarks are passed as input, and machine learning algorithms (Random Forest, XGBoost) are trained to predict the head pose.

- `model_evaluation.py`: 
  - Evaluates the model’s performance using metrics such as Mean Squared Error (MSE) and R² score. It also provides visualizations of predicted poses on the input images.

### **Example Command**

```bash
python main.py
```

## **Key Components**

1. **Dataset Loading**: 
   - The `dataset.py` script loads the images and annotations, converting them into a format suitable for the pose estimation task.
  
2. **Feature Extraction**: 
   - The `image_processing.py` script extracts relevant geometric features from the 3D facial landmarks. Features include distances and angles between key points (e.g., between the eyes, nose, and mouth), which are critical for determining yaw, pitch, and roll.

3. **Pose Estimation**: 
   - In `pose_estimation.py`, regression models (Random Forest and XGBoost) are trained to predict the head pose using the extracted features.

4. **Model Evaluation**: 
   - The `model_evaluation.py` script evaluates the models' predictions on unseen data, calculating error metrics and visualizing results.

## **Results**

- **Evaluation Metrics**: We use Mean Squared Error (MSE) and R² score to evaluate the accuracy of yaw, pitch, and roll predictions.
- **Visual Outputs**: Predicted poses are overlaid on the original images, allowing qualitative assessment.

Here are some sample results:

![result2](https://github.com/user-attachments/assets/e268c942-b4c9-4137-8114-60f10c8b9568)
![result1](https://github.com/user-attachments/assets/bda49138-4f1d-4de2-b885-ad0bbb3bac55)

## **Contributing**

Contributions from the community are highly encouraged. Please feel free to open an issue or submit a pull request if you encounter any bugs, have suggestions, or would like to add new features.

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## **Future Work**

In future updates, I plan to:
- Add support for additional datasets.
- Improve the model by experimenting with deep learning architectures.
- Explore real-time head pose estimation.

## **Contact**

If you have any questions or suggestions, feel free to contact us by opening an issue in this repository.
