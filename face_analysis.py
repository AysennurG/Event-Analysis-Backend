import os
import cv2
import numpy as np
import json
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def get_dominant_from_dict(dct):
    if isinstance(dct, dict) and len(dct) > 0:
        return max(dct, key=dct.get)
    return None

def analyze_and_draw_faces(image_folder, output_folder):
    results = []
    processed_images = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            print(f"Processing file: {filename}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read image {filename}")
                continue

            try:
                analyses = DeepFace.analyze(
                    img_path,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False,
                    detector_backend='retinaface'
                )
                # DeepFace çıktısını JSON'a çevirip tekrar yükle
                analyses = json.loads(json.dumps(analyses, default=str))

                # JSON'dan sonra bile tuple dönebilir, güvenli şekilde düzleştir
                if isinstance(analyses, tuple):
                    flat = []
                    for a in analyses:
                        if a is None:
                            continue
                        if isinstance(a, dict):
                            flat.append(a)
                        elif isinstance(a, tuple):
                            for x in a:
                                if isinstance(x, dict):
                                    flat.append(x)
        # Eğer a tuple değilse, for x in a çalışmaz, hiçbir şey yapma
                    analyses = flat
                elif isinstance(analyses, dict):
                    analyses = [analyses]
                elif not isinstance(analyses, list):
                    analyses = []
            except Exception as e:
                print(f"Error analyzing faces in {filename}: {e}")
                analyses = []

            if any(isinstance(a, dict) for a in analyses):
                for analysis in analyses:
                    if not isinstance(analysis, dict):
                        continue
                    region = analysis.get('region') or analysis.get('facial_area')
                    if not region or not isinstance(region, dict):
                        continue
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    if w == 0 or h == 0:
                        continue

                    face_img = img[y:y+h, x:x+w]
                    if face_img.size == 0:
                        continue

                    try:
                        embedding = DeepFace.represent(
                            face_img,
                            model_name='Facenet',
                            enforce_detection=False,
                            detector_backend='retinaface'
                        )
                        if isinstance(embedding, list):
                            if len(embedding) > 0 and isinstance(embedding[0], dict) and 'embedding' in embedding[0]:
                                embedding = embedding[0]['embedding']
                            elif len(embedding) > 0 and isinstance(embedding[0], (float, int)):
                                pass
                            else:
                                continue
                        elif isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        else:
                            continue
                    except Exception as e:
                        print(f"Embedding error in {filename}: {e}")
                        continue

                    age = analysis.get('age')
                    gender = analysis.get('dominant_gender') or get_dominant_from_dict(analysis.get('gender', {}))
                    race = analysis.get('dominant_race') or get_dominant_from_dict(analysis.get('race', {}))
                    emotion = analysis.get('dominant_emotion') or get_dominant_from_dict(analysis.get('emotion', {}))

                    if age is None or gender is None or race is None:
                        continue

                    results.append({
                        'image': filename,
                        'age': int(age),
                        'gender': gender,
                        'race': race,
                        'emotion': emotion,
                        'embedding': embedding,
                        'coordinates': (int(x), int(y), int(w), int(h))
                    })

                    # Görüntüye etiket ekle
                    label = f"{gender}, {age}, {race}, {emotion}"
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # İşlenmiş resmi kaydet
            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, img)
            processed_images.append(f"processed_{filename}")
            print(f"Processed and saved: {output_path}")

    return results, processed_images

def group_faces_and_generate_report(results, threshold=0.8):
    embeddings = [r['embedding'] for r in results if isinstance(r['embedding'], (list, np.ndarray))]
    grouped_faces = []
    visited = set()
    if embeddings:
        similarity_matrix = cosine_similarity(embeddings)
        for i, result in enumerate(results):
            if i in visited:
                continue
            group = [result]
            visited.add(i)
            for j in range(len(results)):
                if j not in visited and similarity_matrix[i, j] > threshold:
                    group.append(results[j])
                    visited.add(j)
            grouped_faces.append(group)
        crowd_size = len(grouped_faces)
    else:
        crowd_size = 0

    genders = [r['gender'] for r in results]
    races = [r['race'] for r in results]
    ages = [r['age'] for r in results]
    emotions = [r['emotion'] for r in results if r.get('emotion')]

    report = {
        "crowd_size": crowd_size,
        "gender_counts": dict(Counter(genders)),
        "race_counts": dict(Counter(races)),
        "age_counts": dict(Counter(ages)),
        "emotion_counts": dict(Counter(emotions)),
        "ages": ages
    }
    return report