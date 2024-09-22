import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh for landmark extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_face_mesh(image_path):
    """Extract face mesh landmarks from an image."""
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

def draw_axis(img, yaw, pitch, roll, nose_tip, size=100):
    """Draw axes on the image based on yaw, pitch, and roll angles."""
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
