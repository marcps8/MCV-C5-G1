import torch
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_and_crop_face(image_path, model_name='yolov5s', confidence_threshold=0.5):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    model.to(device)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read the image.")
        return

    # Inference
    results = model(image)

    # Results
    detections = results.xyxy[0]  # detections in xyxy format
    max_confidence = 0
    best_box = None

    # Select the face with the highest confidence
    for *xyxy, conf, cls in detections:
        if conf > max_confidence and conf > confidence_threshold:
            max_confidence = conf
            best_box = [int(i) for i in xyxy]

    # Crop the most confident detected face
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        face = image[y1:y2, x1:x2]
        # Resize to (224, 224)
        face = cv2.resize(face, (224, 224))  # Resize the face
        # Replace the original image with the face image
        cv2.imwrite(image_path, face)
        print("Face cropped and saved successfully.")
    else:
        print("No face detected with confidence above the threshold.")

def detect_and_crop_faces_in_subfolders(root_dir, model_name='yolov5s', confidence_threshold=0.5):
    # Iterate through subfolders
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through images in the subfolder
            if subdir=="1" or subdir=="3":
                for filename in os.listdir(subdir_path):
                    image_path = os.path.join(subdir_path, filename)
                    detect_and_crop_face(image_path, model_name, confidence_threshold)

# Usage
root_directory = '/ghome/group01/MCV-C5-G1/Week6/data/train_augmented/'
detect_and_crop_faces_in_subfolders(root_directory)
