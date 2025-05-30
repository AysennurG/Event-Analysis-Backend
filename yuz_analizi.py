import os
import cv2
import numpy as np
from deepface import DeepFace

def analyze_and_draw_faces(image_folder, output_folder):
    results = []
    processed_images = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(image_folder, filename)
            print(f"Processing file: {filename}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read image {filename}")
                continue

            # Görseli büyüt (küçük yüzler için)
            scale = 2
            img = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale))

            try:
                analyses = DeepFace.analyze(
                    img,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False,
                    detector_backend='retinaface'
                )
                if isinstance(analyses, dict):
                    analyses = [analyses]
            except Exception as e:
                print(f"Error analyzing faces in {filename}: {e}")
                analyses = []

            for analysis in analyses:
                try:
                    region = analysis.get('region') or analysis.get('facial_area')
                    if not (region and isinstance(region, dict)):
                        continue
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    if w == 0 or h == 0:
                        continue
                    face_img = img[y:y+h, x:x+w]
                    embedding = DeepFace.represent(
                        face_img,
                        model_name='Facenet',
                        enforce_detection=False,
                        detector_backend='retinaface'
                    )
                    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], dict) and 'embedding' in embedding[0]:
                        embedding = embedding[0]['embedding']
                    elif isinstance(embedding, np.ndarray):
                        pass
                    else:
                        continue

                    results.append({
                        'image': filename,
                        'age': int(analysis['age']),
                        'gender': analysis['dominant_gender'],
                        'race': analysis['dominant_race'],
                        'emotion': analysis['dominant_emotion'],
                        'embedding': embedding,
                        'coordinates': (x, y, w, h)
                    })

                    label = f"{analysis['dominant_gender']}, {analysis['age']}, {analysis['dominant_race']}, {analysis['dominant_emotion']}"
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing face in {filename}: {e}")

            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, img)
            processed_images.append(f"processed_{filename}")
            print(f"Processed and saved: {output_path}")

    return results, processed_images