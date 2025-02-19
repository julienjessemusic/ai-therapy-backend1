from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure OpenAI
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("WARNING: OpenAI API key not found in environment variables!")
else:
    # Print first and last 4 characters of the API key for verification
    print(f"API Key loaded: {api_key[:4]}...{api_key[-4:]}")

client = OpenAI(api_key=api_key)

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
        # Verify API key is set
        if not api_key:
            return jsonify({
                'error': 'OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.',
                'status': 'error'
            }), 500

        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        print(f"Sending request to OpenAI with message: {user_message[:50]}...")
        print(f"Using API key ending in: ...{api_key[-4:]}")
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )

        # Extract the assistant's message
        ai_message = response.choices[0].message.content
        print(f"Received response from OpenAI: {ai_message[:50]}...")

        return jsonify({
            'message': ai_message,
            'status': 'success'
        })

    except OpenAIError as e:
        error_message = str(e)
        print(f"Full OpenAI Error: {error_message}")
        
        if "exceeded your current quota" in error_message:
            error_message = "The OpenAI API key has exceeded its quota. Please check your billing details at https://platform.openai.com/account/billing. You may need to wait a few minutes for billing changes to take effect."
        elif "invalid api key" in error_message.lower():
            error_message = "The OpenAI API key is invalid. Please check your API key at https://platform.openai.com/api-keys"
        
        print(f"OpenAI Error: {error_message}")
        return jsonify({
            'error': error_message,
            'status': 'error'
        }), 500

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
