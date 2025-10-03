from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import secrets

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, created_at, last_login, is_active):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at
        self.last_login = last_login
        self.is_active = is_active

    @staticmethod
    def create(db, username, email, password):
        cursor = db.cursor()
        now = datetime.utcnow().isoformat()
        hashed_password = generate_password_hash(password)
        
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, created_at, is_active) VALUES (?, ?, ?, ?, ?)',
            (username, email, hashed_password, now, True)
        )
        db.commit()
        return cursor.lastrowid

    @staticmethod
    def get_by_id(db, user_id):
        cursor = db.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        if user:
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                password_hash=user['password_hash'],
                created_at=user['created_at'],
                last_login=user['last_login'],
                is_active=user['is_active']
            )
        return None

    @staticmethod
    def get_by_email(db, email):
        cursor = db.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        if user:
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                password_hash=user['password_hash'],
                created_at=user['created_at'],
                last_login=user['last_login'],
                is_active=user['is_active']
            )
        return None

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def create_session(self, db):
        cursor = db.cursor()
        token = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        expires = now + timedelta(days=30)
        
        cursor.execute(
            'INSERT INTO sessions (user_id, session_token, created_at, expires_at) VALUES (?, ?, ?, ?)',
            (self.id, token, now.isoformat(), expires.isoformat())
        )
        db.commit()
        return token

    @staticmethod
    def get_by_session(db, token):
        cursor = db.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.session_token = ? AND s.expires_at > ?
        ''', (token, now))
        user = cursor.fetchone()
        if user:
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                password_hash=user['password_hash'],
                created_at=user['created_at'],
                last_login=user['last_login'],
                is_active=user['is_active']
            )
        return None