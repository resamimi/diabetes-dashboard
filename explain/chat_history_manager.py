from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from typing import List, Tuple, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
import json
import time

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)
    activities = db.relationship('UserActivity', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserActivity(db.Model):
    __tablename__ = 'user_activities'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    activity_type = db.Column(db.String(50))  # e.g., 'time_spent', 'interaction', 'chat'
    component_name = db.Column(db.String(100))  # e.g., 'FeatureRangePlot'
    duration = db.Column(db.Integer)  # Duration in milliseconds
    details = db.Column(db.JSON)  # Store any additional context
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)  # Track active/inactive periods
    
    @staticmethod
    def add_activity(
        user_id: int, 
        activity_type: str, 
        details: Dict[str, Any], 
        component_name: str = None,
        duration: int = None,
        session_id: str = None,
        is_active: bool = True
    ):
        """Enhanced method to create and save a new activity"""
        activity = UserActivity(
            user_id=user_id,
            activity_type=activity_type,
            component_name=component_name,
            duration=duration,
            details=details,
            session_id=session_id,
            is_active=is_active
        )
        db.session.add(activity)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        return activity
    
    @staticmethod
    def get_time_analytics(user_id: int) -> Dict[str, Any]:
        """Get comprehensive time analytics for a user"""
        try:
            # Get all time spent activities
            activities = UserActivity.query.filter_by(
                user_id=user_id,
                activity_type='time_spent'
            ).order_by(UserActivity.timestamp.desc()).all()
            
            # Aggregate by component
            component_times = {}
            session_times = {}
            current_session = None
            
            for activity in activities:
                component = activity.component_name
                duration = activity.duration or 0
                
                # Initialize component stats if needed
                if component not in component_times:
                    component_times[component] = {
                        'total_time': 0,
                        'active_time': 0,
                        'inactive_time': 0,
                        'sessions': set(),
                        'avg_session_duration': 0,
                        'interactions': 0
                    }
                
                # Track times
                component_times[component]['total_time'] += duration
                if activity.is_active:
                    component_times[component]['active_time'] += duration
                else:
                    component_times[component]['inactive_time'] += duration
                
                # Track sessions
                if activity.session_id:
                    component_times[component]['sessions'].add(activity.session_id)
                    
                    # Track session times
                    if activity.session_id != current_session:
                        if current_session and current_session in session_times:
                            session_times[current_session]['end'] = activity.timestamp
                        session_times[activity.session_id] = {
                            'start': activity.timestamp,
                            'component': component
                        }
                        current_session = activity.session_id
            
            # Calculate averages and clean up data for JSON
            for component in component_times:
                sessions = len(component_times[component]['sessions'])
                total_time = component_times[component]['total_time']
                component_times[component]['sessions'] = sessions
                component_times[component]['avg_session_duration'] = (
                    total_time / sessions if sessions > 0 else 0
                )
            
            return {
                'component_times': component_times,
                'session_history': [
                    {
                        'session_id': session_id,
                        'component': data['component'],
                        'start_time': data['start'].isoformat(),
                        'end_time': data.get('end', datetime.utcnow()).isoformat()
                    }
                    for session_id, data in session_times.items()
                ]
            }
            
        except Exception as e:
            print(f"Error getting time analytics: {str(e)}")
            return {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'activity_type': self.activity_type,
            'component_name': self.component_name,
            'duration': self.duration,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'is_active': self.is_active
        }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    response_id = db.Column(db.String(100), nullable=False)
    visualization_data = db.Column(db.JSON, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.String(50), nullable=True)
    session_id = db.Column(db.String(100))
    duration = db.Column(db.Integer)  # Time spent on this chat interaction
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'prompt': self.prompt,
            'response': self.response,
            'response_id': self.response_id,
            'visualization_data': self.visualization_data,
            'timestamp': self.timestamp.isoformat(),
            'feedback': self.feedback,
            'session_id': self.session_id,
            'duration': self.duration
        }

class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    user_agent = db.Column(db.String(500))
    device_info = db.Column(db.JSON)
    total_duration = db.Column(db.Integer)  # Total session duration
    
    @staticmethod
    def start_session(user_id: int, session_id: str, user_agent: str, device_info: Dict, max_retries=3):
        """Start a new session with retry logic."""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # End any existing active sessions
                active_sessions = UserSession.query.filter_by(
                    user_id=user_id,
                    end_time=None
                ).all()
                
                for session in active_sessions:
                    session.end_time = datetime.utcnow()
                    if session.start_time:
                        session.total_duration = int(
                            (datetime.utcnow() - session.start_time).total_seconds() * 1000
                        )
                
                # Create new session
                new_session = UserSession(
                    user_id=user_id,
                    session_id=session_id,
                    user_agent=user_agent,
                    device_info=device_info
                )
                
                db.session.add(new_session)
                db.session.commit()
                return new_session
                
            except Exception as e:
                last_error = e
                retry_count += 1
                db.session.rollback()
                if retry_count == max_retries:
                    raise last_error
                time.sleep(0.1 * retry_count)  # Exponential backoff
    
    def end_session(self):
        """End the current session and calculate duration"""
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.total_duration = int(
                (self.end_time - self.start_time).total_seconds() * 1000
            )
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

class ChatHistoryManager:
    def __init__(self, app=None):
        self.app = app

    def init_app(self, app):
        self.app = app

    def _ensure_app_context(self):
        if self.app is None:
            raise RuntimeError("ChatHistoryManager not initialized with Flask app")

    def get_user_history(self, user_id: str) -> List[Tuple[str, str]]:
        """Get all prompts and responses for a user"""
        with self.app.app_context():
            history = ChatHistory.query.filter_by(user_id=user_id)\
                .order_by(ChatHistory.timestamp.asc())\
                .all()
            return [[entry.prompt, entry.response, entry.visualization_data] for entry in history]
    
    def get_recent_exchanges(self, user_id: str, limit: int = 1) -> List[Tuple[str, str]]:
        """Get the most recent exchanges"""
        with self.app.app_context():
            recent = ChatHistory.query.filter_by(user_id=user_id)\
                .order_by(ChatHistory.timestamp.desc())\
                .limit(limit)\
                .all()
            return [[entry.prompt, entry.response, entry.visualization_data] for entry in reversed(recent)]
            