import psycopg2

# PostgreSQL bağlantısı
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname="event_analysis",
            user="postgres",
            password="ays123",
            host="localhost"
        )
        cursor = conn.cursor()
        print("Database connection successful")
        return conn, cursor
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return None, None

# Tabloları oluşturma
def create_tables():
    conn, cursor = connect_to_db()
    if not conn or not cursor:
        print("Failed to connect to the database")
        return

    try:
        # `users` tablosunu oluştur
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

        # `events` tablosunu oluştur
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

        # `event_images` tablosunu oluştur
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_images (
            id SERIAL PRIMARY KEY,
            event_id INT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            image_path VARCHAR(255) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # `face_analysis_results` tablosunu oluştur
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_analysis_results (
            id SERIAL PRIMARY KEY,
            event_image_id INT NOT NULL REFERENCES event_images(id) ON DELETE CASCADE,
            age INT,
            gender VARCHAR(50),
            race VARCHAR(50),
            embedding JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        print("All tables created successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Tabloları oluşturma fonksiyonunu çağır
if __name__ == "__main__":
    create_tables()