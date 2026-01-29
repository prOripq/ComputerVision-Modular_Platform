import cv2

class VideoLoader:
    def __init__(self, source=0):
        # source может быть цифрой 0 (вебкамера) или ссылкой "rtsp://..."
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть источник видео: {source}")

    def get_frame(self):
        # Читаем один кадр
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        # Освобождаем камеру при выходе
        self.cap.release()