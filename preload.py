"""Preload bot initialization for the Flask app."""
from explain.logic import ExplainBot
import os
import gin
from logging.config import dictConfig
from flask import Blueprint
import time 
import psutil

# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl

def preload_bot():

    try:

        print("Pre-loading model and data...")
        start_time = time.time()

        # Parse gin global config
        gin.parse_config_file("global_config.gin")

        # Get args
        args = GlobalArgs()

        bp = Blueprint('host', __name__, template_folder='templates')

        dictConfig({
            'version': 1,
            'formatters': {'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            }},
            'handlers': {'wsgi': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'default'
            }},
            'root': {
                'level': 'INFO',
                'handlers': ['wsgi']
            }
        })

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Parse application level configs
        gin.parse_config_file(args.config)

        bot = ExplainBot()

        end_time = time.time()
        print(f"Pre-loading completed in {end_time - start_time:.2f} seconds")

        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        return bot, bp, args
        
    except Exception as e:
        print(f"Pre-load failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Initialize bot globally
# initialized_bot, bp, args = preload_bot()