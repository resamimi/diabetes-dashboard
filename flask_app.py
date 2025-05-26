"""The app main with chat history functionality."""
import json
import logging
from logging.config import dictConfig
import os
import traceback
import requests
import uuid
from flask import Flask, request, Blueprint, jsonify, send_from_directory
import gin
import copy
from explain.logic import ExplainBot
from explain.sample_prompts_by_action import sample_prompt_for_action
from datetime import datetime, timedelta

from explain.chat_history_manager import db, ChatHistory, User, UserActivity, UserSession

from preload import preload_bot
from flask_sqlalchemy import SQLAlchemy

print("Starting flask_app.py module load")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir()}")

# Initialize these at module level
initialized_bot = None
bp = None
args = None
app = None

try:
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load bot and blueprint
    print("Starting preload")
    initialized_bot, bp, args = preload_bot()
    print("Preload completed")

    # Define all routes BEFORE creating the app
    # Health check endpoint
    @bp.route('/health')
    def health_check():
        try:
            print("Health check requested")
            return 'OK', 200
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return 'Error', 500

    def safe_activity_log(user_id: int, activity_type: str, details: dict) -> None:
        """Safely log user activity without raising exceptions"""
        try:
            # Verify user exists
            user = User.query.get(user_id)
            if not user:
                print(f"Warning: Attempted to log activity for non-existent user {user_id}")
                return

            UserActivity.add_activity(
                user_id=user_id,
                activity_type=activity_type,
                details=details
            )
            db.session.commit()
        except Exception as e:
            print(f"Warning: Failed to log activity: {str(e)}")
            db.session.rollback()

    @bp.route('/api/auth/signup', methods=['POST'])
    def signup():
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({'error': 'Username and password are required'}), 400
                
            if User.query.filter_by(username=username).first():
                return jsonify({'error': 'Username already exists'}), 400
                
            # Create new user
            user = User(username=username)
            user.set_password(password)
            
            # Add and commit user first
            db.session.add(user)
            db.session.commit()
            
            # Get the newly created user's ID
            user_id = user.id
            
            # Create initial session
            session_id = str(uuid.uuid4())
            user_agent = request.headers.get('User-Agent')
            device_info = {
                'ip': request.remote_addr,
                'platform': request.user_agent.platform,
                'browser': request.user_agent.browser,
                'version': request.user_agent.version
            }
            
            try:
                # Create session
                session = UserSession(
                    user_id=user_id,
                    session_id=session_id,
                    user_agent=user_agent,
                    device_info=device_info
                )
                db.session.add(session)
                db.session.commit()
                
                # Use safe_activity_log for signup activity
                safe_activity_log(
                    user_id=user_id,
                    activity_type='user_signup',
                    details={
                        'timestamp': datetime.utcnow().isoformat(),
                        'session_id': session_id,
                        'device_info': device_info
                    }
                )
                
                return jsonify({
                    'id': user_id,
                    'username': username,
                    'session_id': session_id
                })
                    
            except Exception as session_error:
                print(f"Session creation failed: {str(session_error)}")
                db.session.rollback()
                return jsonify({'error': 'Failed to create user session'}), 500
                
        except Exception as e:
            print(f"Signup error: {str(e)}")
            db.session.rollback()
            return jsonify({'error': 'Error creating user'}), 500

    @bp.route('/api/auth/signin', methods=['POST'])
    def signin():
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            print(f"Signin attempt for username: {username}")
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                # Create a new session
                session_id = str(uuid.uuid4())
                user_agent = request.headers.get('User-Agent')
                device_info = {
                    'ip': request.remote_addr,
                    'platform': request.user_agent.platform,
                    'browser': request.user_agent.browser,
                    'version': request.user_agent.version
                }
                
                try:
                    # Create session
                    session = UserSession(
                        user_id=user.id,
                        session_id=session_id,
                        user_agent=user_agent,
                        device_info=device_info
                    )
                    db.session.add(session)
                    db.session.commit()
                    
                    # Use safe_activity_log for signin activity
                    safe_activity_log(
                        user_id=user.id,
                        activity_type='user_signin',
                        details={
                            'timestamp': datetime.utcnow().isoformat(),
                            'session_id': session_id,
                            'device_info': device_info
                        }
                    )
                    
                    print(f"Successful signin for user: {username}")
                    return jsonify({
                        'id': user.id,
                        'username': user.username,
                        'session_id': session_id
                    })
                    
                except Exception as session_error:
                    print(f"Session creation failed: {str(session_error)}")
                    db.session.rollback()
                    return jsonify({'error': 'Failed to create user session'}), 500

            else:
                print(f"Failed signin attempt for username: {username}")
                return jsonify({'error': 'Invalid username or password'}), 401
            
        except Exception as e:
            print(f"Signin error: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            db.session.rollback()
            return jsonify({'error': 'Error during signin'}), 500

    @bp.route('/api/auth/signout', methods=['POST'])
    def signout():
        try:
            data = request.get_json()
            session_id = data.get('sessionId')
            user_id = data.get('userId')
            
            if not user_id:
                return jsonify({'error': 'User ID is required'}), 400
                
            # Convert user_id to integer
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid user ID format'}), 400
                
            # Verify user exists
            user = User.query.get(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
                
            if session_id:
                # End the user session
                session = UserSession.query.filter_by(
                    session_id=session_id,
                    user_id=user_id,
                    end_time=None
                ).first()
                
                if session:
                    session.end_time = datetime.utcnow()
                    db.session.commit()
            
            # Use safe_activity_log for signout activity
            safe_activity_log(
                user_id=user_id,
                activity_type='user_signout',
                details={
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': session_id
                }
            )
            
            return jsonify({'message': 'Successfully signed out'})
        
        except Exception as e:
            print(f"Signout error: {str(e)}")
            db.session.rollback()
            return jsonify({'error': 'Error during sign out'}), 500
                

    @bp.route('/api/activity/log', methods=['POST'])
    def log_activity():
        """Log user activity"""
        try:
            data = request.get_json()
            user_id = data.get('userId')
            activity_type = data.get('type')
            details = data.get('details', {})
            
            if not user_id:
                return jsonify({'error': 'User ID is required'}), 400
                
            # Convert user_id to int if it's a string
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid user ID format'}), 400
                
            # Get or create session ID from details
            session_id = details.get('sessionId', str(uuid.uuid4()))
            details['session_id'] = session_id
            
            # Use safe_activity_log helper
            safe_activity_log(user_id, activity_type, details)
            
            return jsonify({'message': 'Activity logged successfully'})
            
        except Exception as e:
            print(f"Activity logging error: {str(e)}")
            return jsonify({'error': 'Error logging activity'}), 500

    @bp.route('/api/analytics/user/<user_id>', methods=['GET'])
    def get_user_analytics(user_id):
        """Get analytics for a specific user"""
        try:
            # Total activities
            total_activities = UserActivity.query.filter_by(user_id=user_id).count()
            
            # Chat interactions
            chat_interactions = UserActivity.query.filter_by(
                user_id=user_id,
                activity_type='chat_interaction'
            ).count()
            
            # Visualization interactions
            viz_interactions = db.session.query(
                UserActivity.details['visualizationType'].label('viz_type'),
                db.func.count().label('count')
            ).filter(
                UserActivity.user_id == user_id,
                UserActivity.activity_type == 'visualization_interaction'
            ).group_by('viz_type').all()
            
            # Session data
            sessions = UserSession.query.filter_by(user_id=user_id).all()
            avg_session_duration = db.session.query(
                db.func.avg(
                    db.func.extract('epoch', UserSession.end_time - UserSession.start_time)
                )
            ).filter(
                UserSession.user_id == user_id,
                UserSession.end_time.isnot(None)
            ).scalar()
            
            return jsonify({
                'total_activities': total_activities,
                'chat_interactions': chat_interactions,
                'visualization_interactions': [
                    {'type': viz[0], 'count': viz[1]} 
                    for viz in viz_interactions
                ],
                'total_sessions': len(sessions),
                'avg_session_duration_seconds': float(avg_session_duration) if avg_session_duration else None
            })
            
        except Exception as e:
            print(f"Error getting analytics: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/analytics/time-spent/<user_id>', methods=['GET'])
    def get_time_analytics(user_id):
        try:
            # Use the enhanced analytics method from UserActivity
            analytics = UserActivity.get_time_analytics(user_id)
            
            if not analytics:
                return jsonify({
                    "error": "No activity data found for user"
                }), 404
                
            return jsonify(analytics)
            
        except Exception as e:
            print(f"Error getting time analytics: {str(e)}")
            return jsonify({
                "error": "Failed to retrieve analytics",
                "details": str(e)
            }), 500

    @bp.route('/api/analytics/export/<user_id>', methods=['GET'])
    def export_user_data(user_id):
        """Export all user activity data"""
        try:
            # Get all activities
            activities = UserActivity.query.filter_by(user_id=user_id).all()
            
            # Get all chat history
            chats = ChatHistory.query.filter_by(user_id=user_id).all()
            
            # Get all sessions
            sessions = UserSession.query.filter_by(user_id=user_id).all()
            
            export_data = {
                'activities': [activity.to_dict() for activity in activities],
                'chat_history': [chat.to_dict() for chat in chats],
                'sessions': [
                    {
                        'session_id': session.session_id,
                        'start_time': session.start_time.isoformat(),
                        'end_time': session.end_time.isoformat() if session.end_time else None,
                        'user_agent': session.user_agent,
                        'device_info': session.device_info
                    }
                    for session in sessions
                ]
            }
            
            return jsonify(export_data)
            
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/data')
    def get_data():
        """API endpoint to get initial data for React"""
        try:
            objective = app.bot.conversation.describe.get_dataset_objective()
            return jsonify({
                "currentUserId": "user",
                "datasetObjective": objective
            })
        except Exception as e:
            print(f"Error in get_data: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/chat_history/<user_id>")
    def get_chat_history(user_id):
        """Get chat history for a specific user."""
        try:
            history = ChatHistory.query.filter_by(user_id=user_id)\
                .order_by(ChatHistory.timestamp.desc())\
                .limit(100)\
                .all()
            return jsonify([entry.to_dict() for entry in history])
        except Exception as e:
            print(f"Error fetching chat history: {str(e)}")
            return jsonify({"error": "Failed to fetch chat history"}), 500

    @bp.route("/api/conversations/recent/<user_id>")
    def get_recent_conversations(user_id):
        """Get recent conversations for a user."""
        try:
            recent = ChatHistory.query.filter_by(user_id=user_id)\
                .order_by(ChatHistory.timestamp.desc())\
                .limit(10)\
                .all()
            
            conversations = []
            for entry in recent:
                conversations.append({
                    'prompt': entry.prompt,
                    'response': entry.response,
                    'timestamp': entry.timestamp.isoformat(),
                    'has_visualization': entry.visualization_data is not None
                })
            
            return jsonify(conversations)
        except Exception as e:
            print(f"Error getting recent conversations: {str(e)}")
            return jsonify({"error": "Failed to fetch conversations"}), 500

    @bp.route("/log_feedback", methods=['POST'])
    def log_feedback():
        """Logs feedback and updates chat history."""
        try:
            feedback = request.data.decode("utf-8")
            print(feedback)
            split_feedback = feedback.split(" || ")

            message = f"Feedback formatted improperly. Got: {split_feedback}"
            assert split_feedback[0].startswith("MessageID: "), message
            assert split_feedback[1].startswith("Feedback: "), message
            assert split_feedback[2].startswith("Username: "), message

            message_id = split_feedback[0][len("MessageID: "):]
            feedback_text = split_feedback[1][len("Feedback: "):]
            username = split_feedback[2][len("Username: "):]

            # Update chat history with feedback
            chat_entry = ChatHistory.query.filter_by(
                response_id=message_id,
                user_id=username
            ).first()
            
            if chat_entry:
                chat_entry.feedback = feedback_text
                db.session.commit()

            logging_info = {
                "id": message_id,
                "feedback_text": feedback_text,
                "username": username
            }
            app.bot.log(logging_info)
            return "", 204
        except Exception as e:
            print(f"Error logging feedback: {str(e)}")
            return jsonify({"error": "Failed to log feedback"}), 500

    @bp.route("/sample_prompt", methods=["POST"])
    def sample_prompt():
        """Samples a prompt"""
        try:
            data = json.loads(request.data)
            action = data["action"]
            username = data["thisUserName"]

            prompt = sample_prompt_for_action(
                action,
                app.bot.prompts.filename_to_prompt_id,
                app.bot.prompts.final_prompt_set,
                real_ids=app.bot.conversation.get_training_data_ids()
            )

            logging_info = {
                "username": username,
                "requested_action_generation": action,
                "generated_prompt": prompt
            }
            app.bot.log(logging_info)

            return prompt
        except Exception as e:
            print(f"Error sampling prompt: {str(e)}")
            return jsonify({"error": "Failed to sample prompt"}), 500

    @bp.route("/get_bot_response", methods=['POST'])
    def get_bot_response():
        if request.method == "POST":
            print("Entering get_bot_response endpoint")
            try:
                data = json.loads(request.data)
                user_text = data["userInput"]
                user_id = data.get("userId")
                
                if not user_id:
                    return jsonify({"error": "User ID is required"}), 400
                    
                print(f"Processing request for user: {user_id}, text: {user_text}")
                
                # Verify conversation object exists
                if not hasattr(app.bot, 'conversation'):
                    print("Bot conversation object not found")
                    raise ValueError("Bot conversation not initialized")
                
                conversation = app.bot.conversation
                conversation.username = str(user_id)  # Use user_id as username
                
                response = app.bot.update_state(user_text, conversation)
                
                # Split response and visualization data
                msg_text = response
                visualization_data = None
                timestamp = datetime.utcnow()
                
                if '<json>' in response:
                    parts = response.split('<json>')
                    msg_text = parts[0]
                    json_str = parts[1].split("<>")[0]
                    try:
                        visualization_data = json.loads(json_str)
                    except json.JSONDecodeError as json_err:
                        print(f"JSON parsing error: {json_err}")
                        visualization_data = None
                
                # Generate unique response ID
                response_id = str(id(response))
                
                try:
                    # Save to database with user_id
                    chat_entry = ChatHistory(
                        user_id=user_id,
                        prompt=user_text,
                        response=msg_text,
                        response_id=response_id,
                        visualization_data=visualization_data,
                        timestamp=timestamp
                    )
                    db.session.add(chat_entry)
                    db.session.commit()
                    print("Successfully saved to database")
                    
                except Exception as db_error:
                    print(f"Database error: {str(db_error)}")
                    # Continue even if database save fails
                
                return f"{response}<>{response_id}"
                
            except Exception as e:
                print(f"Error in get_bot_response: {str(e)}")
                print(f"Full traceback: {traceback.format_exc()}")
                response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
                return f"{response}<>{str(id(response))}"

    @bp.route('/api/patient/<patient_id>', methods=['GET'])
    def get_patient_data(patient_id):
        """Get patient data endpoint."""
        try:
            patient_data = app.bot.get_patient_data(patient_id)
            return jsonify(patient_data)
        except Exception as e:
            print(f"Error getting patient data: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/prediction/<patient_id>', methods=['GET'])
    def get_prediction(patient_id):
        """Get prediction endpoint."""
        try:
            prediction = app.bot.get_prediction(patient_id)
            response = {
                "result": "High Risk" if prediction['prediction'] == 1 else "Low Risk",
                "confidence": round(prediction['probability'] * 100, 1)
            }
            return jsonify(response)
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/visualization/<viz_type>/<patient_id>', methods=['GET'])
    def get_visualization_data(viz_type, patient_id):
        """Get visualization data endpoint."""
        try:
            viz_data = app.bot.get_visualization_data(viz_type, patient_id)
            return jsonify(viz_data)
        except Exception as e:
            print(f"Error getting visualization: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/translate', methods=['POST'])
    def translate_text():
        """Proxy endpoint for Azure Translator"""
        try:
            print("Received translation request")
            # Log the request data
            data = request.json
            print(f"Request data: {data}")
            text = data.get('text')
            target_lang = data.get('to')  # Note: changed from 'target_lang' to 'to'

            if not text or not target_lang:
                print("Missing required parameters")
                return jsonify({'error': 'Missing text or target language'}), 400
            # Log the Azure Translator credentials (first few characters)
            subscription_key = os.environ.get('AZURE_TRANSLATOR_KEY')
            if not subscription_key:
                print("Azure Translator key not found in environment variables")
                return jsonify({'error': 'API key not configured'}), 500
                
            print(f"Using Azure Translator key: {subscription_key[:5]}...")
            # Azure Translator configuration
            endpoint = os.environ.get('AZURE_TRANSLATOR_ENDPOINT', 'https://api.cognitive.microsofttranslator.com')
            location = os.environ.get('AZURE_TRANSLATOR_REGION', 'global')
            # Construct Azure Translator API request
            path = '/translate'
            constructed_url = endpoint + path

            # Prepare request parameters and headers
            params = {
                'api-version': '3.0',
                'from': 'en',
                'to': target_lang
            }
            headers = {
                'Ocp-Apim-Subscription-Key': subscription_key,
                'Ocp-Apim-Subscription-Region': location,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }
            # Prepare request body
            body = [{
                'text': text
            }]
            print(f"Sending request to Azure Translator: {constructed_url}")
            print(f"Request params: {params}")
            print(f"Request headers: {headers}")
            print(f"Request body: {body}")

            try:
                # Make the request
                azure_response = requests.post(
                    constructed_url, 
                    params=params, 
                    headers=headers, 
                    json=body
                )
                # Log the response
                print(f"Azure Translator response status: {azure_response.status_code}")
                print(f"Azure Translator response content: {azure_response.text}")
                # Check for request errors
                azure_response.raise_for_status()
                # Parse response
                translation_response = azure_response.json()
                translation = translation_response[0]['translations'][0]['text']

                print(f"Translated text: {translation}")

                return jsonify({
                    'translation': translation
                })
            except requests.RequestException as req_error:
                # Log the full error details
                error_msg = f"Azure Translation Request Error: {str(req_error)}"
                print(error_msg)
                            # Log response content if available
                if hasattr(req_error, 'response'):
                    print(f"Response content: {req_error.response.text}")
                    
                return jsonify({
                    'error': error_msg
                }), 500
        except Exception as e:
            # Catch-all for any other unexpected errors
            error_msg = f"Unexpected translation error: {str(e)}"
            print(error_msg)
            print(f"Full traceback: {traceback.format_exc()}")
            return jsonify({
                'error': error_msg
            }), 500

    @bp.route("/api/what-if/<patient_id>", methods=['POST'])
    def get_what_if_prediction(patient_id):
        """Get prediction for modified patient values."""
        try:

            # Get the modified values from request
            new_values = request.json
            prediction = app.bot.get_what_if_prediction(patient_id, new_values)

            # Format response with both prediction and confidence
            response = {
                "result": "High Risk" if prediction['prediction'] == 1 else "Low Risk",
                "confidence": round(prediction['probability'] * 100, 1)  # Convert to percentage
            }
            
            return jsonify(response)

            return jsonify(prediction)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Serve React App
    @bp.route('/', defaults={'path': ''})
    @bp.route('/<path:path>')
    def serve(path):
        """Serve React App"""
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    # NOW create the app after all routes are defined
    # Update create_app function
    def create_app():
        print("Entering create_app")
        app = Flask(__name__, static_folder='static/react/chat-interface/build', static_url_path='')
        
        # Database configuration
        database_url = os.environ.get("DATABASE_URL")
        if database_url is None:
            print("WARNING: No DATABASE_URL found, falling back to SQLite")
            database_url = 'sqlite:///chat_history.db'
        elif database_url.startswith("postgres://"):
            print("Converting postgres:// to postgresql://")
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        # Add SSL configuration to the database URL
        if database_url.startswith('postgresql://'):
            if '?' not in database_url:
                database_url += '?'
            else:
                database_url += '&'
            database_url += 'sslmode=require'
            
        print(f"Using database URL: {database_url[:database_url.find('@') if '@' in database_url else None]}")
        
        # Configure SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_pre_ping': True,  # Enable connection health checks
            'pool_recycle': 300,    # Recycle connections every 5 minutes
            'pool_timeout': 30,     # Connection timeout after 30 seconds
            'pool_size': 10,        # Maximum number of connections
            'max_overflow': 5       # Maximum number of connections that can be created beyond pool_size
        }
        
        # Initialize extensions
        db.init_app(app)
        
        # Register blueprint
        app.register_blueprint(bp, url_prefix=args.baseurl)
        print(f"Blueprint registered with baseurl: {args.baseurl}")
        
        # Create tables if they don't exist
        with app.app_context():
            try:
                db.create_all()
                print("Database tables verified")
            except Exception as e:
                print(f"Error creating tables: {str(e)}")
                # Fall back to SQLite if PostgreSQL connection fails
                if database_url.startswith('postgresql://'):
                    print("Falling back to SQLite database")
                    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
                    db.create_all()
        
        # Initialize bot
        app.bot = initialized_bot
        app.bot.history_manager.init_app(app)
        print("Bot and history manager initialized")
        
        return app

    # Create the application instance
    print("Creating app instance")
    app = create_app()
    print("App instance created")

except Exception as e:
    print(f"Error during application initialization: {str(e)}")
    traceback.print_exc()
    raise

# Configure Gunicorn logging if not running directly
if __name__ != '__main__':
    print("Setting up gunicorn logging")
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    print("Gunicorn configuration complete")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Running in __main__ with port {port}")
    app.run(host="0.0.0.0", port=port)
else:
    print("Not running in __main__")

print(f"Module load complete. App variable type: {type(app)}")
print("Flask application ready to serve requests")