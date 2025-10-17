from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, role='technician'):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)
    
    @staticmethod
    def get_by_id(db, user_id):
        """Get user by ID"""
        user_data = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user_data:
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                role=user_data.get('role', 'technician')
            )
        return None
    
    @staticmethod
    def get_by_email(db, email):
        """Get user by email"""
        user_data = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user_data:
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                role=user_data.get('role', 'technician')
            )
        return None
    
    @staticmethod
    def create(db, username, email, password, role='technician'):
        """Create new user with hashed password"""
        password_hash = generate_password_hash(password)
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)',
            (username, email, password_hash, role)
        )
        db.commit()
        return cursor.lastrowid