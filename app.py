from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
from collections import Counter
import psycopg2
import os
import shutil
import traceback
import zipfile
import requests as http_requests
import threading
from db import connect_to_db

if not os.getenv("HF_SPACE_URL"):
    from face_analysis import analyze_and_draw_faces, group_faces_and_generate_report


def _generate_report(results):
    if not results:
        return {}
    genders = Counter(r.get("gender") for r in results if r.get("gender"))
    races = Counter(r.get("race") for r in results if r.get("race"))
    emotions = Counter(r.get("emotion") for r in results if r.get("emotion"))
    ages = Counter(str(r.get("age")) for r in results if r.get("age"))
    happy = emotions.get("happy", 0) + emotions.get("surprise", 0)
    satisfaction = round(happy / len(results) * 100, 1) if results else 0
    return {
        "crowd_size": len(results),
        "gender_distribution": dict(genders),
        "race_distribution": dict(races),
        "emotion_distribution": dict(emotions),
        "age_distribution": dict(ages),
        "memnuniyet_orani_%": satisfaction
    }


def make_report(results):
    hf_url = os.getenv("HF_SPACE_URL")
    if hf_url:
        return _generate_report(results)
    return group_faces_and_generate_report(results)


def call_hf_space(image_folder, hf_url):
    # HF Space'i uyandır
    try:
        http_requests.get(f"{hf_url}/health", timeout=30)
    except Exception:
        pass

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
                    print(f"HF response for {filename}: {data}")
                    for face in data.get("faces", []):
                        face["image"] = filename
                        results.append(face)
                else:
                    print(f"HF error {resp.status_code} for {filename}: {resp.text}")
            except Exception as e:
                print(f"HF Space error for {filename}: {e}")
    return results


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(app, supports_credentials=True, origins=[
    frontend_url,
    "http://localhost:3000",
    r"https://.*\.vercel\.app",
])

app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_DOMAIN"] = None

# PostgreSQL bağlantısı
conn, cursor = connect_to_db()

# Tabloları oluştur
if conn and cursor:
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255),
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255),
                google_id VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                event_name VARCHAR(255) NOT NULL,
                event_date TIMESTAMP NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_images (
                id SERIAL PRIMARY KEY,
                event_id INT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                image_path VARCHAR(255) NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_analysis_results (
                id SERIAL PRIMARY KEY,
                event_image_id INT NOT NULL REFERENCES event_images(id) ON DELETE CASCADE,
                age INT,
                gender VARCHAR(50),
                race VARCHAR(50),
                emotion VARCHAR(64),
                embedding JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("Tables ready")
    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {str(e)}")


def get_db():
    global conn, cursor
    try:
        if conn is None or conn.closed:
            raise Exception("closed")
        conn.cursor().execute("SELECT 1")
    except Exception:
        conn, cursor = connect_to_db()
    return conn, cursor


@app.before_request
def refresh_db():
    global conn, cursor
    conn, cursor = get_db()


login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    try:
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if user:
            return User(user[0], user[1], user[2])
    except Exception:
        pass
    return None


google_blueprint = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="google_login",
    redirect_url="/auth/google/callback"
)
app.register_blueprint(google_blueprint, url_prefix="/auth")


@app.route('/auth/google/callback')
def google_login():
    try:
        if not google.authorized:
            return jsonify({"error": "Not authorized"}), 401

        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return jsonify({"error": "Failed to fetch user info from Google"}), 500
        user_info = resp.json()

        email = user_info.get("email")
        google_id = user_info.get("id")
        username = user_info.get("name", "Google User")

        if not email or not google_id:
            return jsonify({"error": "Invalid user info from Google"}), 500

        conn, cursor = get_db()
        cursor.execute("SELECT * FROM users WHERE email = %s OR google_id = %s", (email, google_id))
        user = cursor.fetchone()
        if not user:
            cursor.execute(
                "INSERT INTO users (username, email, google_id) VALUES (%s, %s, %s)",
                (username, email, google_id)
            )
            conn.commit()

        session["user"] = email
        return redirect(os.getenv("FRONTEND_URL", "http://localhost:3000") + "/?oauth=success")
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Eksik bilgi"}), 400

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and user[3] and check_password_hash(user[3], password):
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
        try:
            cursor.execute("SELECT id, username, email FROM users WHERE email = %s", (session["user"],))
            user = cursor.fetchone()
            if user:
                return jsonify({"authenticated": True, "user": {"id": user[0], "username": user[1], "email": user[2]}}), 200
        except Exception:
            pass
    return jsonify({"authenticated": False}), 401


@app.route('/print_results', methods=['POST'])
def print_results():
    try:
        results = request.json.get("results", [])
        for result in results:
            print(f"Resim: {result.get('image')} | Yaş: {result.get('age')} | Cinsiyet: {result.get('gender')} | Irk: {result.get('race')}")
        return jsonify({"message": "Sonuçlar terminale yazdırıldı."}), 200
    except Exception as e:
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
                zip_path = os.path.join(temp_folder, file.filename)
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            zip_info.filename = os.path.basename(zip_info.filename)
                            extracted_path = zip_ref.extract(zip_info, temp_folder)
                            cursor.execute("""
                                INSERT INTO event_images (event_id, image_path)
                                VALUES (%s, %s) RETURNING id
                            """, (event_id, os.path.basename(extracted_path)))
                            event_image_id = cursor.fetchone()[0]
                            conn.commit()
                            image_id_map[os.path.basename(extracted_path)] = event_image_id
            else:
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

        def run_analysis(temp_folder, image_id_map, event_id):
            try:
                bg_conn, bg_cursor = connect_to_db()
                if not bg_conn:
                    print("Background DB connection failed")
                    return

                hf_url = os.getenv("HF_SPACE_URL")
                if hf_url:
                    results = call_hf_space(temp_folder, hf_url)
                else:
                    output_folder = os.path.join(temp_folder, "processed")
                    results, _ = analyze_and_draw_faces(temp_folder, output_folder)

                print(f"Background analysis done. Results: {len(results)}")
                print(f"image_id_map keys: {list(image_id_map.keys())}")

                for result in results:
                    image_name = result.get('image')
                    event_image_id = image_id_map.get(image_name)
                    if not event_image_id:
                        print(f"No match for image: {image_name}")
                        continue
                    bg_cursor.execute("""
                        INSERT INTO face_analysis_results (event_image_id, age, gender, race, emotion)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        event_image_id,
                        result.get('age'),
                        result.get('gender'),
                        result.get('race'),
                        result.get('emotion')
                    ))
                bg_conn.commit()
                print(f"Analysis saved for event {event_id}")
            except Exception as e:
                print(f"Background analysis error: {e}")
                print(traceback.format_exc())
            finally:
                shutil.rmtree(temp_folder, ignore_errors=True)
                try:
                    bg_conn.close()
                except Exception:
                    pass

        thread = threading.Thread(target=run_analysis, args=(temp_folder, image_id_map, event_id))
        thread.daemon = True
        thread.start()

        return jsonify({
            "event_id": event_id,
            "analysis_results": [],
            "report": {},
            "status": "processing"
        }), 200
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
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
                age, gender, race, emotion = face[0], face[1], face[2], face[3] if len(face) > 3 else None
                results.append({
                    "event_image_id": event_image_id,
                    "image": image_path,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "emotion": emotion
                })

        report = make_report(results)

        return jsonify({
            "id": event[0],
            "event_name": event[1],
            "event_date": str(event[2]),
            "description": event[3],
            "results": results,
            "report": report
        }), 200
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(traceback.format_exc())
        return jsonify({"error": "Failed to fetch event details"}), 500


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
                "event_date": str(e[2]),
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
        cursor.execute("""
            DELETE FROM face_analysis_results
            WHERE event_image_id IN (
                SELECT id FROM event_images WHERE event_id = %s
            )
        """, (event_id,))
        cursor.execute("DELETE FROM event_images WHERE event_id = %s", (event_id,))
        cursor.execute("DELETE FROM events WHERE id = %s", (event_id,))
        conn.commit()
        return jsonify({"message": "Etkinlik silindi"}), 200
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print("Etkinlik silinirken hata:", e)
        return jsonify({"error": "Etkinlik silinemedi"}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
