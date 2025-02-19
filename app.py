from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OpenAI API key is not set!")
openai.api_key = api_key

# Enhanced system message for more authentic therapy simulation
SYSTEM_MESSAGE = """You are Dr. Sarah Matthews..."""  # (Keep the rest of the system message)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        logger.info(f"Received request at {datetime.now()}")
        
        user_message = data.get('message')
        message_history = data.get('history', [])
        
        if not user_message:
            logger.error("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400

        # Construct messages with history and enhanced context
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]
        
        # Add message history
        for msg in message_history:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
            
        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Create chat completion with streaming
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            presence_penalty=0.6,
            frequency_penalty=0.5,
            stream=True
        )

        def generate():
            collected_message = ""
            for chunk in response:
                if chunk and chunk.choices and chunk.choices[0].delta.get("content"):
                    content = chunk.choices[0].delta.content
                    collected_message += content
                    # Send each chunk with SSE format
                    yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
            
            # Send final message with done flag
            yield f"data: {json.dumps({'content': '', 'done': True, 'fullMessage': collected_message})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({
            'error': 'OpenAI service error. Please try again.',
            'status': 'error'
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
