import sys
import os
import time
import numpy as np

# Add the root directory to the sys.path to allow imports from core/modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_loader import VideoLoader
from modules.person_module import PersonDetector
from modules.face_module import FaceRecognizer

# --- CONFIGURATION ---
TEST_VIDEO = "video.mp4"  # Path to the test video
NUM_FRAMES_TO_TEST = 100                # Number of frames to process for the test

def run_benchmark():
    print(f"--- RUNNING BENCHMARK ON {NUM_FRAMES_TO_TEST} FRAMES ---")
    
    loader = VideoLoader(TEST_VIDEO)
    person_tracker = PersonDetector()
    face_recognizer = FaceRecognizer(db_path="../known_faces")

    times_yolo = []
    times_face = []
    times_total = []

    frames_processed = 0

    print("Progress: [", end="", flush=True)

    while frames_processed < NUM_FRAMES_TO_TEST:
        frame = loader.get_frame()
        if frame is None:
            break

        start_total = time.perf_counter()

        # 1. Measure YOLO (Person Detection)
        start_yolo = time.perf_counter()
        frame, _ = person_tracker.process(frame)
        times_yolo.append(time.perf_counter() - start_yolo)

        # 2. Measure InsightFace (Face Recognition)
        start_face = time.perf_counter()
        frame, _ = face_recognizer.recognize(frame)
        times_face.append(time.perf_counter() - start_face)

        times_total.append(time.perf_counter() - start_total)
        frames_processed += 1

        if frames_processed % 10 == 0:
            print("█", end="", flush=True)

    print("]\n")
    loader.release()

    # --- RESULTS ANALYSIS ---
    avg_yolo = np.mean(times_yolo) * 1000  # Convert to milliseconds
    avg_face = np.mean(times_face) * 1000
    avg_total = np.mean(times_total) * 1000
    avg_fps = 1.0 / np.mean(times_total)

    print("========== BENCHMARK RESULTS ==========")
    print(f"Frames Processed:           {frames_processed}")
    print(f"Average Pipeline FPS:       {avg_fps:.1f} FPS")
    print("-" * 39)
    print(f"Avg YOLOv8 Inference Time:  {avg_yolo:.2f} ms/frame")
    print(f"Avg InsightFace Time:       {avg_face:.2f} ms/frame")
    print(f"Total Processing Time:      {avg_total:.2f} ms/frame")
    print("=======================================")

if __name__ == "__main__":
    run_benchmark()