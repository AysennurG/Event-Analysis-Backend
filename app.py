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
from db import connect_to_db
from face_analysis import analyze_and_draw_faces, group_faces_and_generate_report

# .env dosyasını yükle
load_dotenv()

# Flask uygulaması
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
CORS(app, supports_credentials=True)

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
        return redirect("http://127.0.0.1:3000/")  # Frontend ana sayfasına yönlendir
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and check_password_hash(user[3], password):  # Şifre doğrulama
        user_obj = User(user[0], user[1], user[2])
        login_user(user_obj)
        session["user"] = email
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"error": "Invalid email or password"}), 401

@app.route('/logout')
@login_required
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
            file_path = os.path.join(temp_folder, file.filename)
            file.save(file_path)
            # Her dosya için event_images tablosuna kayıt ekle
            cursor.execute("""
                INSERT INTO event_images (event_id, image_path)
                VALUES (%s, %s) RETURNING id
            """, (event_id, file.filename))
            event_image_id = cursor.fetchone()[0]
            conn.commit()
            image_id_map[file.filename] = event_image_id

        output_folder = os.path.join(temp_folder, "processed")
        results, processed_images = analyze_and_draw_faces(temp_folder, output_folder)

        report = group_faces_and_generate_report(results)

        analysis_results = []
        for result in results:
            image_name = result.get('image')
            event_image_id = image_id_map.get(image_name)
            cursor.execute("""
                INSERT INTO face_analysis_results (event_image_id, age, gender, race, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                event_image_id,
                result.get('age'),
                result.get('gender'),
                result.get('race'),
                str(result.get('embedding'))
            ))
            conn.commit()
            analysis_results.append({
                "event_image_id": event_image_id,
                "image": image_name,
                "age": result.get('age'),
                "gender": result.get('gender'),
                "race": result.get('race'),
                "emotion": result.get('emotion')
            })

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
                age, gender, race, emotion = face
                results.append({
                    "event_image_id": event_image_id,
                    "image": image_path,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "emotion": emotion
                })

        return jsonify({
            "event_details": {
                "id": event[0],
                "event_name": event[1],
                "event_date": event[2],
                "description": event[3],
                "results": results
            }
        }), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Failed to fetch event details"}), 500

if __name__ == '__main__':
    app.run(debug=True)