from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
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
client = OpenAI(api_key=api_key)

# Enhanced system message for more authentic therapy simulation
SYSTEM_MESSAGE = """You are Dr. Sarah Matthews, a highly experienced clinical psychologist with over 15 years of practice. You specialize in integrative therapy, combining various evidence-based approaches to provide personalized care. Your therapeutic style is characterized by:

PROFESSIONAL DEMEANOR:
1. Warm yet Professional Presence:
   - Use a calm, measured tone
   - Maintain professional boundaries while being approachable
   - Express genuine empathy and understanding
   - Speak in a clear, thoughtful manner

2. Therapeutic Process:
   - Begin sessions with "How have you been since our last conversation?"
   - Take time to explore emotions and experiences deeply
   - Use strategic silence when appropriate
   - Guide rather than direct the conversation

3. Clinical Expertise:
   - Draw from multiple therapeutic modalities:
     * Cognitive Behavioral Therapy (CBT)
     * Psychodynamic approaches
     * Mindfulness-based techniques
     * Solution-focused strategies
   - Adapt approach based on client's needs
   - Recognize patterns in thinking and behavior
   - Make connections between past and present experiences"""

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

        # Create chat completion with streaming using the new API
        stream = client.chat.completions.create(
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
            for chunk in stream:
                if chunk.choices[0].delta.content:
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

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
