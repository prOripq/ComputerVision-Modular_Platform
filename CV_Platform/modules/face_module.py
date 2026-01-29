import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self, db_path='known_faces'):
        print("Загрузка модели распознавания лиц (InsightFace)...")
        # buffalo_l - это мощная модель, которая включает детекцию и распознавание
        self.app = FaceAnalysis(name='buffalo_l')
        # ctx_id=0 использовать видеокарту, -1 - процессор. 
        # det_size - размер картинки для детектора (640x640 стандарт)
        self.app.prepare(ctx_id=0, det_size=(640, 640)) 
        
        self.known_embeddings = [] # Тут храним цифровые слепки лиц
        self.known_names = []      # Тут храним имена
        
        # Загружаем лица из папки
        self.load_database(db_path)

    def load_database(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Папка {path} создана. Положите туда фото для распознавания.")
            return

        print("Обучение на фотографиях из папки...")
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                filepath = os.path.join(path, filename)
                img = cv2.imread(filepath)
                if img is None: continue
                
                # Ищем лицо на фото
                faces = self.app.get(img)
                
                if len(faces) > 0:
                    # Берем первое найденное лицо и сохраняем его "код" (embedding)
                    # normed_embedding - это уже готовый вектор для сравнения
                    embedding = faces[0].normed_embedding
                    name = filename.split('.')[0] # убираем расширение файла
                    
                    self.known_embeddings.append(embedding)
                    self.known_names.append(name)
                    print(f"Загружен пользователь: {name}")
                else:
                    print(f"В файле {filename} лица не найдены.")

    def process(self, frame):
        # 1. Ищем все лица на кадре с камеры
        faces = self.app.get(frame)
        
        # 2. Пробегаемся по каждому найденному лицу
        for face in faces:
            # Получаем координаты лица (bbox)
            box = face.bbox.astype(int)
            color = (0, 0, 255) # Красный (неизвестный)
            name = "Unknown"
            
            # 3. Сравниваем с нашей базой
            if self.known_embeddings:
                # Считаем схожесть (dot product) текущего лица со всеми в базе
                # Чем больше число, тем больше похоже.
                current_emb = face.normed_embedding
                sims = np.dot(self.known_embeddings, current_emb)
                
                # Находим индекс самого похожего лица
                best_idx = np.argmax(sims)
                score = sims[best_idx]
                
                # Если похожесть больше 0.5 (50%), считаем что узнали
                if score > 0.5:
                    name = f"{self.known_names[best_idx]} ({int(score*100)}%)"
                    color = (0, 255, 0) # Зеленый (свой)

            # 4. Рисуем рамку и имя
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame
    
    def process_and_return_names(self, frame):
        """
        То же самое, что process, но возвращает еще и список имен.
        """
        faces = self.app.get(frame)
        found_names = []
        
        for face in faces:
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            name = "Unknown"
            
            if self.known_embeddings:
                current_emb = face.normed_embedding
                sims = np.dot(self.known_embeddings, current_emb)
                best_idx = np.argmax(sims)
                score = sims[best_idx]
                
                if score > 0.5:
                    name_full = self.known_names[best_idx]
                    name = name_full # Просто имя для списка
                    label = f"{name_full} ({int(score*100)}%)"
                    color = (0, 255, 0)
                    found_names.append(name) # <-- Добавляем в список
                else:
                    label = name

            # Рисуем
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name if name == "Unknown" else label, 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame, found_names