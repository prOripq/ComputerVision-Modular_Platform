import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.face_module import FaceRecognizer

# --- CONFIGURATION ---
# Test images directory structure:
# test_dataset_faces/
# ├── admin/
# │   ├── photo1.jpg
# ├── unknown/
# │   ├── stranger1.jpg
TEST_DIR = "test_dataset_faces"

def evaluate_face_accuracy():
    print("--- EVALUATING FACE RECOGNITION ACCURACY ---")
    
    # Initialize the recognizer (it will load known_faces from the root folder)
    recognizer = FaceRecognizer(db_path="../known_faces")
    
    total_images = 0
    correct_predictions = 0
    false_positives = 0  # Unknown classified as a known person
    false_negatives = 0  # Known person classified as Unknown

    for true_label in os.listdir(TEST_DIR):
        person_dir = os.path.join(TEST_DIR, true_label)
        if not os.path.isdir(person_dir): continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath)
            if img is None: continue

            total_images += 1
            
            # Run inference
            _, found_names = recognizer.recognize(img)
            
            predicted_label = found_names[0] if found_names else "Unknown"

            if predicted_label == true_label:
                correct_predictions += 1
            else:
                if true_label == "Unknown" and predicted_label != "Unknown":
                    false_positives += 1
                elif true_label != "Unknown" and predicted_label == "Unknown":
                    false_negatives += 1
                
                print(f"[MISMATCH] File: {filename} | Expected: {true_label} | Got: {predicted_label}")

    if total_images == 0:
        print(f"No images found in {TEST_DIR}.")
        return

    accuracy = (correct_predictions / total_images) * 100

    print("\n========== FACE RECOGNITION METRICS ==========")
    print(f"Total Test Images: {total_images}")
    print(f"Correct Matches:   {correct_predictions}")
    print(f"False Positives:   {false_positives}")
    print(f"False Negatives:   {false_negatives}")
    print("-" * 46)
    print(f"Overall Accuracy:  {accuracy:.1f}%")
    print("==============================================")

if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        print(f"Please create the '{TEST_DIR}' folder and populate it with test images.")
    else:
        evaluate_face_accuracy()