from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self):
        print("Загрузка нейросети YOLOv8...")
        # 'n' значит nano. Самая маленькая и быстрая модель.
        # При первом запуске она скачается сама.
        self.model = YOLO('yolov8n.pt') 
        print("Нейросеть готова к работе.")

    def process(self, frame):
        """
        Принимает кадр, находит людей, рисует рамки.
        Возвращает: обработанный кадр, количество людей.
        """
        # track - это команда "следить" (присваивать ID)
        # persist=True - важно! Чтобы нейросеть помнила объекты с предыдущего кадра
        # classes=0 - искать ТОЛЬКО людей (0 - это код человека в базе COCO)
        # verbose=False - чтобы не спамить в консоль кучей текста
        results = self.model.track(frame, persist=True, classes=0, verbose=False, device=0, half=True)

        # results[0].plot() сам рисует красивые рамки и ID
        annotated_frame = results[0].plot()

        # Считаем, сколько коробочек (boxes) нашла сеть
        people_count = len(results[0].boxes)

        return annotated_frame, people_count