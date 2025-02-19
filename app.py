from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Create chat completion using the client
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

        return jsonify({
            'message': ai_message,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Failed to process message',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
