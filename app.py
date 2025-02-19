from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
import logging
from datetime import datetime

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
        
        # Add message history without timestamps
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
        
        # Create chat completion with enhanced therapeutic context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            presence_penalty=0.6,  # Encourage more diverse responses
            frequency_penalty=0.5   # Reduce repetition
        )

        # Extract and log the assistant's message
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
