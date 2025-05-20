from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
import psycopg2
import os

# .env dosyasını yükle
load_dotenv()

# Flask uygulaması
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
CORS(app, supports_credentials=True)

# PostgreSQL bağlantısı
conn = psycopg2.connect(
    dbname="event_analysis",
    user="postgres",
    password="ays123",
    host="localhost"
)
cursor = conn.cursor()

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

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and check_password_hash(user[2], password):
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

@app.route('/auth/check')
def auth_check():
    if "user" in session:
        print(f"Authenticated user: {session['user']}")  # Oturum bilgisi kontrolü için log ekleyin
        return jsonify({"authenticated": True}), 200
    print("Unauthorized access attempt")  # Yetkisiz erişim kontrolü için log ekleyin
    return jsonify({"authenticated": False}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data['email']
    password = data['password']
    password_hash = generate_password_hash(password)

    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            (email, password_hash)
        )
        conn.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "Email already exists"}), 400
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

# Google OAuth yapılandırması
google_blueprint = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="google_login"  # Callback endpoint'ini özelleştiriyoruz
)
app.register_blueprint(google_blueprint, url_prefix="/auth/google")

@app.route('/auth/google/callback')
def google_login():
    try:
        if not google.authorized:
            return redirect(url_for("google.login"))

        resp = google.get("/oauth2/v2/userinfo")
        user_info = resp.json()
        email = user_info["email"]

        # Kullanıcıyı veritabanında kontrol et veya kaydet
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (%s, %s)",
                (user_info["name"], email)
            )
            conn.commit()

        # Kullanıcı oturumunu işaretle
        session["user"] = email
        return redirect("http://127.0.0.1:3000/")  # Frontend ana sayfasına yönlendir
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)