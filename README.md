# Face Pose Estimation from 2D Images

## Overview

This repository contains a Python implementation for estimating the yaw, pitch, and roll of human faces in images using a 3D facial landmark detection approach. The project utilizes a subset of the AFLW2000-3D dataset, specifically 64 images annotated with corresponding .mat files. By leveraging these annotations, we aim to build a robust model that accurately predicts head poses from 2D images, even in challenging orientations.

### Idea Behind the Project

The ability to estimate head poses is essential for various applications, including human-computer interaction, virtual reality, and facial recognition. Traditional 2D face detection techniques often struggle with poses that deviate significantly from frontal views. This project tackles the challenge by using 3D facial landmarks, allowing the model to better understand the spatial orientation of a face. 

### Methodology

1. **Dataset Usage**: A subset of the AFLW2000-3D dataset is critical for training and evaluating our models. Each of the 64 images is annotated with 68 facial landmarks, which provide key information about the face's geometry and pose.

2. **Feature Extraction**: The model computes various geometric features from the landmarks, such as distances and angles between key points. These features serve as input for our machine learning algorithms, which predict the head pose.

3. **Model Training**: We use several regression models, including Random Forest and XGBoost, to predict yaw, pitch, and roll. Each model is trained on the extracted features and evaluated based on its accuracy in predicting head poses.

4. **Evaluation Metrics**: Model performance is assessed using metrics like Mean Squared Error (MSE) and R² score, providing insight into the effectiveness of the predictions.

## Dataset Reference

### AFLW2000-3D

The AFLW2000-3D dataset was introduced by Zhu et al. in their paper titled "Face Alignment Across Large Poses: A 3D Solution." This dataset contains 2000 images annotated with 68-point 3D facial landmarks. In this project, we use a subset of 64 images along with their corresponding .mat files for pose estimation.

- **Citation**:
  ```
  Zhu, X., Wang, Z., & Liu, Y. (2016). Face Alignment Across Large Poses: A 3D Solution. In European Conference on Computer Vision (ECCV).
  ```

## Installation

To run the code in this repository, you need to install the following packages:

```bash
pip install -r requirements.txt
```

Ensure that you have Python 3.6 or later installed.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayatollah-Ibrahim/Head_Pose_Estimation.git
   cd Head_Pose_Estimation
   ```

2. **Prepare the Data**:
   Place your 64 annotated images and their corresponding .mat files in the designated `data/JPEG` and `data/MAT` directories, respectively. Ensure the structure looks like this:
   ```
   data/
       JPEG/
           image1.jpg
           image2.jpg
           ...
       MAT/
           image1.mat
           image2.mat
           ...
   ```

3. **Run the Code**:
   You can execute the main script using:
   ```bash
   python main.py
   ```

4. **Input and Output**:
   The script will load images and annotations, extract features, train models, and display images with predicted head poses overlaid. Make sure to monitor the console for any output or error messages during execution.

## Code Structure

- `main.py`: Main script for face pose estimation.
- `dataset.py`: Handles dataset creation and preprocessing.
- `image_processing.py`: Contains functions for image processing and feature extraction.
- `model_evaluation.py`: Responsible for evaluating model performance.
- `pose_estimation.py`: Contains functions for pose estimation using trained models.
- `requirements.txt`: List of required packages.
- `README.md`: This documentation file.

## Detailed Explanation of Key Components

### 1. Dataset Creation

The `create_dataset` function in `dataset.py` is responsible for loading images and their corresponding landmark annotations from the specified directories. It extracts the 68 facial landmarks from each image and organizes the data for further processing.

### 2. Feature Extraction

The implementation computes various geometric features based on the distances and angles between selected facial landmarks. These features are crucial for accurately estimating yaw, pitch, and roll.

### 3. Model Training

The project utilizes regression models like Random Forest and XGBoost, which are trained using the features extracted from the landmarks. Hyperparameters are tuned to optimize performance, ensuring the best possible predictions.

### 4. Evaluation

Model evaluation is conducted in `model_evaluation.py`, where metrics like Mean Squared Error (MSE) and R² score are calculated to assess the accuracy of predictions on the test data.

## Results

The results from model training and evaluation will be displayed in the terminal. Visualizations of the predicted head poses overlaid on the input images will also be generated.

### Example Output

Below are examples of the output images with predicted head poses:

![result1](https://github.com/user-attachments/assets/0a17406e-52ae-43ba-a33a-178bf70e53e0)
![result2](https://github.com/user-attachments/assets/e2c3a5ec-f224-4c15-b154-c8ac15d42190)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or additional features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
