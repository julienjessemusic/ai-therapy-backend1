from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
import logging

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

# System message to guide the AI's responses
SYSTEM_MESSAGE = """You are a supportive AI therapy assistant. While you're not a replacement for a licensed therapist:
- Respond with empathy and understanding
- Help users explore their thoughts and feelings
- Encourage positive coping strategies
- NEVER give medical advice
- If someone is in crisis, direct them to emergency services
- Maintain a professional, caring tone
- Keep responses concise but meaningful"""

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        logger.info(f"Received request: {data}")
        
        user_message = data.get('message')
        message_history = data.get('history', [])
        
        if not user_message:
            logger.error("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400

        # Construct messages with history
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

        logger.info(f"Sending request to OpenAI with message history length: {len(messages)}")
        
        # Create chat completion using the older API format
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Changed back to gpt-3.5-turbo
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        # Extract the assistant's message
        ai_message = response.choices[0].message['content']
        logger.info(f"Received response from OpenAI: {ai_message[:100]}...")

        return jsonify({
            'message': ai_message,
            'status': 'success'
        })

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
