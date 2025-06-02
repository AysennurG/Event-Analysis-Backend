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

    detector_backend = 'mtcnn'

    total_faces = 0
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, filename)
        print(f"\nProcessing file: {filename}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {filename}, skipping...")
            continue

        try:
            analyses = DeepFace.analyze(
                img_path,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=True,
                detector_backend=detector_backend
            )
        except Exception as e:
            print(f"Error analyzing faces in {filename}: {e}")
            continue

        if isinstance(analyses, dict):
            analyses = [analyses]
        elif not isinstance(analyses, list):
            continue

        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue

            region = analysis.get('region') or analysis.get('facial_area')
            if isinstance(region, dict):
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            elif isinstance(region, (list, tuple)) and len(region) == 4:
                x, y, w, h = region
            else:
                x, y, w, h = 0, 0, img.shape[1], img.shape[0]

            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            age = analysis.get('age')
            gender = analysis.get('dominant_gender') or get_dominant_from_dict(analysis.get('gender', {}))
            race = analysis.get('dominant_race') or get_dominant_from_dict(analysis.get('race', {}))
            emotion = analysis.get('dominant_emotion') or get_dominant_from_dict(analysis.get('emotion', {}))

            print(f"Analysis result for {filename}:")
            print({
                'age': age,
                'gender': gender,
                'race': race,
                'emotion': emotion,
                'region': region
            })

            if age is None or gender is None or race is None:
                continue

            results.append({
                'image': filename,
                'age': int(age),
                'gender': gender,
                'race': race,
                'emotion': emotion,
                'coordinates': (int(x), int(y), int(w), int(h))
            })

            label = f"{gender}, {age}, {race}, {emotion}"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            total_faces += 1

        output_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(output_path, img)
        processed_images.append(f"processed_{filename}")
        print(f"Processed and saved: {output_path}")

    print(f"\nTotal faces analyzed successfully: {total_faces}")
    return results, processed_images

def group_faces_and_generate_report(results, threshold=0.7):
    from collections import Counter
    import numpy as np
    import json

    # Eğer hiç sonuç yoksa, boş bir rapor dön
    if not results or len(results) == 0:
        report = {
            "crowd_size": 0,
            "gender_distribution": {},
            "race_distribution": {},
            "age_distribution": {},
            "emotion_distribution": {},
            "memnuniyet_orani_%": 0
        }
        print("\n--- Rapor (Boş) ---")
        print(json.dumps(report, indent=4, ensure_ascii=False))
        return report

    # Eğer embedding ile gruplayacaksan buraya ekleyebilirsin, ama çoğu zaman gerek yok
    crowd_size = len(results)
    genders = [r['gender'] for r in results]
    races = [r['race'] for r in results]
    ages = [r['age'] for r in results]
    emotions = [r['emotion'] for r in results if r.get('emotion')]

    memnuniyet = Counter(emotions).get("happy", 0) + Counter(emotions).get("surprise", 0)
    total_faces = len(emotions)
    memnuniyet_orani = round((memnuniyet / total_faces) * 100, 2) if total_faces > 0 else 0

    report = {
        "crowd_size": crowd_size,
        "gender_distribution": dict(Counter(genders)),
        "race_distribution": dict(Counter(races)),
        "age_distribution": dict(Counter(ages)),
        "emotion_distribution": dict(Counter(emotions)),
        "memnuniyet_orani_%": memnuniyet_orani
    }

    print("\n--- Rapor ---")
    print(json.dumps(report, indent=4, ensure_ascii=False))
    return report