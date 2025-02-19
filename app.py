from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai
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
    raise ValueError("OpenAI API key is not set in environment variables!")

# Initialize the OpenAI client (fixed initialization)
client = openai.Client(api_key=api_key)

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
        logger.info("Received request")
        
        user_message = data.get('message')
        if not user_message:
            logger.error("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400

        # Create chat completion using the client instance
        logger.info("Sending request to OpenAI")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )

        # Extract the assistant's message
        ai_message = response.choices[0].message.content
        logger.info("Received response from OpenAI")

        return jsonify({
            'message': ai_message,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
