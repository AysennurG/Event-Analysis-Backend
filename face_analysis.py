import os
import cv2
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

def analyze_and_draw_faces(image_folder, output_folder):
    """
    Yüzleri analiz eder, işlenmiş resimleri kaydeder ve analiz sonuçlarını döndürür.

    Args:
        image_folder (str): Yüklenen resimlerin bulunduğu klasör.
        output_folder (str): İşlenmiş resimlerin kaydedileceği klasör.

    Returns:
        tuple: Analiz sonuçları ve işlenmiş resimlerin dosya adları.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]

                try:
                    # Yüz analizi ve embedding çıkarma
                    analysis = DeepFace.analyze(face_img, actions=['age', 'gender', 'race'], enforce_detection=False)
                    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)

                    if isinstance(embedding, list) and len(embedding) > 0:
                        embedding = embedding[0]['embedding']

                    results.append({
                        'image': filename,
                        'age': int(analysis[0]['age']),
                        'gender': analysis[0]['dominant_gender'],
                        'race': analysis[0]['dominant_race'],
                        'embedding': embedding,
                        'coordinates': (int(x), int(y), int(w), int(h))
                    })

                    # Görüntüye etiket ekleme
                    label = f"{analysis[0]['dominant_gender']}, {analysis[0]['age']}, {analysis[0]['dominant_race']}"
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error analyzing face in {filename}: {e}")

            # İşlenmiş resmi kaydet
            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, img)
            processed_images.append(f"processed_{filename}")
            print(f"Processed and saved: {output_path}")

    return results, processed_images


def group_faces_and_generate_report(results, threshold=0.6):
    """
    Yüzleri gruplandırır ve bir rapor oluşturur.

    Args:
        results (list): Yüz analizi sonuçları.
        threshold (float): Benzerlik eşik değeri.

    Returns:
        str: Oluşturulan raporun dosya yolu.
    """
    if not results:
        print("No faces detected. Cannot generate report.")
        return

    embeddings = np.array([result['embedding'] for result in results if isinstance(result['embedding'], list)])

    if len(embeddings) == 0:
        print("No valid embeddings found. Cannot generate report.")
        return

    # Benzerlik matrisini hesapla
    similarity_matrix = cosine_similarity(embeddings)
    grouped_faces = []
    visited = set()

    for i, result in enumerate(results):
        if i in visited:
            continue
        group = [result]
        visited.add(i)
        for j in range(len(results)):
            if j not in visited and similarity_matrix[i, j] > (1 - threshold):
                group.append(results[j])
                visited.add(j)
        grouped_faces.append(group)

    print(f"Total unique faces: {len(grouped_faces)}")

    # Analiz verilerini toplama
    genders = [result['gender'] for result in results]
    races = [result['race'] for result in results]
    ages = [result['age'] for result in results]

    # Grafik oluşturma
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    gender_counts = Counter(genders)
    plt.bar(gender_counts.keys(), gender_counts.values(), color='skyblue')
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    race_counts = Counter(races)
    plt.bar(race_counts.keys(), race_counts.values(), color='lightgreen')
    plt.title("Race Distribution")
    plt.xlabel("Race")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(ages, bins=10, color='salmon', edgecolor='black')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")

    # Raporu kaydet
    graph_path = os.path.join("uploads", "report.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return graph_path