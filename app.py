from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
import psycopg2
import os
import shutil
import traceback
import json
import zipfile
import requests as http_requests
from db import connect_to_db
from face_analysis import analyze_and_draw_faces, group_faces_and_generate_report


def call_hf_space(image_folder, hf_url):
    results = []
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(image_folder, filename)
        with open(img_path, 'rb') as f:
            try:
                resp = http_requests.post(
                    f"{hf_url}/analyze",
                    files={"image": (filename, f, "image/jpeg")},
                    timeout=120
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for face in data.get("faces", []):
                        face["image"] = filename
                        results.append(face)
            except Exception as e:
                print(f"HF Space error for {filename}: {e}")
    return results

# .env dosyasını yükle
load_dotenv()

# Flask uygulaması
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
CORS(app, supports_credentials=True)

# Oturum çerezi ayarları
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_ENV") == "production"

# PostgreSQL bağlantısı
conn, cursor = connect_to_db()

# Flask-Login ayarları
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    if user:
        return User(user[0], user[1], user[2])
    return None

# Google OAuth yapılandırması
google_blueprint = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="google_login"
)
app.register_blueprint(google_blueprint, url_prefix="/auth/google")

@app.route('/auth/google/callback')
def google_login():
    try:
        if not google.authorized:
            return redirect(url_for("google.login"))

        # Google'dan kullanıcı bilgilerini al
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return jsonify({"error": "Failed to fetch user info from Google"}), 500
        user_info = resp.json()

        email = user_info.get("email")
        google_id = user_info.get("id")
        username = user_info.get("name", "Google User")

        if not email or not google_id:
            return jsonify({"error": "Invalid user info from Google"}), 500

        # Kullanıcıyı veritabanında kontrol et veya kaydet
        cursor.execute("SELECT * FROM users WHERE email = %s OR google_id = %s", (email, google_id))
        user = cursor.fetchone()
        if not user:
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, google_id) VALUES (%s, %s, %s)",
                    (username, email, google_id)
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                return jsonify({"error": f"Database error: {str(e)}"}), 500

        # Kullanıcı oturumunu işaretle
        session["user"] = email
        return redirect(os.getenv("FRONTEND_URL", "http://127.0.0.1:3000") + "/")
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    print("Gelen login JSON:", data)  # DEBUG
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Eksik bilgi"}), 400

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and check_password_hash(user[3], password):  # Şifre doğrulama
        user_obj = User(user[0], user[1], user[2])
        login_user(user_obj)
        session["user"] = email
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"error": "Invalid email or password"}), 401

@app.route('/logout', methods=['GET'])
def logout():
    logout_user()
    session.pop("user", None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/auth/check', methods=['GET'])
def auth_check():
    if "user" in session:
        cursor.execute("SELECT id, username, email FROM users WHERE email = %s", (session["user"],))
        user = cursor.fetchone()
        if user:
            return jsonify({"authenticated": True, "user": {"id": user[0], "username": user[1], "email": user[2]}}), 200
    return jsonify({"authenticated": False}), 401

@app.route('/print_results', methods=['POST'])
def print_results():
    try:
        results = request.json.get("results", [])
        print("\n--- Analiz Sonuçları (Butona Tıklanınca) ---")
        for result in results:
            print(f"Resim: {result.get('image')} | Yaş: {result.get('age')} | Cinsiyet: {result.get('gender')} | Irk: {result.get('race')}")
        print("--- Son ---\n")
        return jsonify({"message": "Sonuçlar terminale yazdırıldı."}), 200
    except Exception as e:
        print("Sonuçları terminale yazdırırken hata:", e)
        return jsonify({"error": "Terminale yazdırılamadı."}), 500

@app.route('/upload', methods=['POST'])
def upload_photo():
    try:
        event_name = request.form.get("event_name")
        if not event_name:
            return jsonify({"error": "Event name is required"}), 400

        user_email = session.get("user")
        if not user_email:
            return jsonify({"error": "Unauthorized"}), 401

        cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 403
        user_id = user[0]

        cursor.execute("""
            INSERT INTO events (user_id, event_name, event_date, description)
            VALUES (%s, %s, NOW(), %s) RETURNING id
        """, (user_id, event_name, "Uploaded via analysis"))
        event_id = cursor.fetchone()[0]
        conn.commit()

        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({"error": "No selected file"}), 400

        temp_folder = f"temp_upload_{event_id}"
        os.makedirs(temp_folder, exist_ok=True)

        image_id_map = {}

        for file in files:
            if file.filename.endswith('.zip'):
                # Zip dosyasını aç
                zip_path = os.path.join(temp_folder, file.filename)
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        # Sadece resim dosyalarını al (.jpg, .jpeg, .png)
                        if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Sadece dosya adını kullan, alt klasörleri yok say
                            zip_info.filename = os.path.basename(zip_info.filename)
                            extracted_path = zip_ref.extract(zip_info, temp_folder)
                            cursor.execute("""
                                INSERT INTO event_images (event_id, image_path)
                                VALUES (%s, %s) RETURNING id
                            """, (event_id, os.path.basename(extracted_path)))
                            event_image_id = cursor.fetchone()[0]
                            conn.commit()
                            image_id_map[os.path.basename(extracted_path)] = event_image_id
                        # Uygun olmayan dosyaları atla
            else:
                # Tekil resim dosyası ise
                if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(temp_folder, file.filename)
                    file.save(file_path)
                    cursor.execute("""
                        INSERT INTO event_images (event_id, image_path)
                        VALUES (%s, %s) RETURNING id
                    """, (event_id, file.filename))
                    event_image_id = cursor.fetchone()[0]
                    conn.commit()
                    image_id_map[file.filename] = event_image_id
                # Uygun olmayan dosyaları atla

        output_folder = os.path.join(temp_folder, "processed")
        hf_url = os.getenv("HF_SPACE_URL")
        if hf_url:
            results = call_hf_space(temp_folder, hf_url)
            processed_images = []
        else:
            results, processed_images = analyze_and_draw_faces(temp_folder, output_folder)
        report = group_faces_and_generate_report(results)

        analysis_results = []
        for result in results:
            image_name = result.get('image')
            event_image_id = image_id_map.get(image_name)
            if not event_image_id:
                continue  # Eşleşmeyen dosyaları atla
            cursor.execute("""
                INSERT INTO face_analysis_results (event_image_id, age, gender, race, emotion)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                event_image_id,
                result.get('age'),
                result.get('gender'),
                result.get('race'),
                result.get('emotion')
            ))
            analysis_results.append({
                "event_image_id": event_image_id,
                "image": image_name,
                "age": result.get('age'),
                "gender": result.get('gender'),
                "race": result.get('race'),
                "emotion": result.get('emotion')
            })
        conn.commit()

        shutil.rmtree(temp_folder)

        return jsonify({
            "event_id": event_id,
            "analysis_results": analysis_results,
            "report": report
        }), 200
    except Exception as e:
        conn.rollback()
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to upload photo: {str(e)}"}), 500

@app.route('/event/<int:event_id>', methods=['GET'])
def get_event_details(event_id):
    try:
        cursor.execute("SELECT id, event_name, event_date, description FROM events WHERE id = %s", (event_id,))
        event = cursor.fetchone()
        if not event:
            return jsonify({"error": "Event not found"}), 404

        cursor.execute("SELECT id, image_path FROM event_images WHERE event_id = %s", (event_id,))
        images = cursor.fetchall()

        results = []
        for image in images:
            event_image_id, image_path = image
            cursor.execute("""
                SELECT age, gender, race, emotion FROM face_analysis_results
                WHERE event_image_id = %s
            """, (event_image_id,))
            faces = cursor.fetchall()
            for face in faces:
                if len(face) == 4:
                    age, gender, race, emotion = face
                else:
                    age, gender, race = face
                    emotion = None
                results.append({
                    "event_image_id": event_image_id,
                    "image": image_path,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "emotion": emotion
                })

        report = group_faces_and_generate_report(results)

        return jsonify({
            "id": event[0],
            "event_name": event[1],
            "event_date": event[2],
            "description": event[3],
            "results": results,
            "report": report
        }), 200
    except Exception as e:
        conn.rollback()
        print(traceback.format_exc())
        return jsonify({"error": "Failed to fetch event details"}), 500

# app.py içine ekle
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Eksik bilgi"}), 400

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        return jsonify({"error": "Bu email zaten kayıtlı"}), 409

    hashed_pw = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
        (username, email, hashed_pw)
    )
    conn.commit()
    return jsonify({"message": "Kayıt başarılı"}), 200

@app.route('/events', methods=['GET'])
def get_events():
    try:
        user_email = session.get("user")
        if not user_email:
            return jsonify({"error": "Unauthorized"}), 401

        cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 403
        user_id = user[0]

        cursor.execute("""
            SELECT id, event_name, event_date, description
            FROM events
            WHERE user_id = %s
            ORDER BY event_date DESC
        """, (user_id,))
        events = cursor.fetchall()
        event_list = [
            {
                "id": e[0],
                "event_name": e[1],
                "event_date": e[2],
                "description": e[3]
            }
            for e in events
        ]
        return jsonify({"events": event_list}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to fetch events"}), 500

@app.route('/events/<int:event_id>', methods=['DELETE'])
def delete_event(event_id):
    try:
        # Önce etkinliğe ait yüz analiz sonuçlarını sil
        cursor.execute("""
            DELETE FROM face_analysis_results
            WHERE event_image_id IN (
                SELECT id FROM event_images WHERE event_id = %s
            )
        """, (event_id,))
        # Sonra etkinliğe ait resimleri sil
        cursor.execute("DELETE FROM event_images WHERE event_id = %s", (event_id,))
        # Son olarak etkinliği sil
        cursor.execute("DELETE FROM events WHERE id = %s", (event_id,))
        conn.commit()
        return jsonify({"message": "Etkinlik silindi"}), 200
    except Exception as e:
        conn.rollback()
        print("Etkinlik silinirken hata:", e)
        return jsonify({"error": "Etkinlik silinemedi"}), 500

# face_analysis_results tablosuna emotion sütunu ekle
try:
    cursor.execute("""
        ALTER TABLE face_analysis_results
        ADD COLUMN emotion VARCHAR(64)
    """)
    conn.commit()
except Exception as e:
    conn.rollback()
    print(f"Error adding column to face_analysis_results: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)