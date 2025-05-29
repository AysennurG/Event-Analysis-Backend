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
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST")
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

@app.route('/auth/check')
def auth_check():
    if "user" in session:
        return jsonify({"authenticated": True, "user": session["user"]}), 200
    return jsonify({"authenticated": False}), 401

if __name__ == '__main__':
    app.run(debug=True)