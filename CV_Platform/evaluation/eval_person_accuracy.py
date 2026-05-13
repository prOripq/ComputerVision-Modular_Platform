import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.person_module import PersonDetector

# --- CONFIGURATION ---
# Test images directory structure:
# test_dataset_people/
# ├── 0/          # Images with 0 people
# ├── 1/          # Images with 1 person
# ├── 3/          # Images with 3 people
TEST_DIR = "test_dataset_people"

def evaluate_person_detection():
    print("--- EVALUATING PERSON DETECTION ACCURACY ---")
    detector = PersonDetector()
    
    total_images = 0
    exact_matches = 0
    errors = []

    for folder_name in os.listdir(TEST_DIR):
        folder_path = os.path.join(TEST_DIR, folder_name)
        if not os.path.isdir(folder_path): continue
        
        try:
            # The folder name represents the ground truth number of people
            ground_truth = int(folder_name)
        except ValueError:
            print(f"[WARNING] Skipping folder '{folder_name}' - name must be an integer.")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)
            if img is None: continue

            total_images += 1
            
            # Run inference. process() returns (annotated_frame, people_count)
            _, predicted_count = detector.process(img)
            
            # Calculate absolute error for this image
            error = abs(predicted_count - ground_truth)
            errors.append(error)

            if predicted_count == ground_truth:
                exact_matches += 1
            else:
                print(f"[MISMATCH] File: {filename} | Expected: {ground_truth} | Got: {predicted_count}")

    if total_images == 0:
        print(f"No valid images found in {TEST_DIR}.")
        return

    accuracy = (exact_matches / total_images) * 100
    mae = np.mean(errors)

    print("\n========== PERSON DETECTION METRICS ==========")
    print(f"Total Test Images:   {total_images}")
    print(f"Exact Matches:       {exact_matches}")
    print("-" * 46)
    print(f"Exact Match Accuracy: {accuracy:.1f}%")
    print(f"Mean Absolute Error:  {mae:.2f} people/image")
    print("==============================================")

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"Please create the '{TEST_DIR}' folder and populate it with test images.")
    else:
        evaluate_person_detection()