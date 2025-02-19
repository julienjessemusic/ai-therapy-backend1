from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
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
    print(f"API Key loaded: {api_key[:4]}...{api_key[-4:]}")

openai.api_key = api_key

# Enhanced system message with specific therapeutic approaches
SYSTEM_MESSAGE = """You are an empathetic AI therapy assistant trained in multiple therapeutic modalities. While you're not a replacement for a licensed therapist, you utilize evidence-based approaches including:

THERAPEUTIC APPROACH:
1. Cognitive Behavioral Therapy (CBT):
   - Help identify negative thought patterns
   - Guide users to challenge cognitive distortions
   - Encourage behavioral activation
   - Use Socratic questioning to promote self-discovery

2. Person-Centered Therapy:
   - Practice unconditional positive regard
   - Show genuine empathy and warmth
   - Help clients achieve self-actualization
   - Follow the client's lead in discussions

3. Solution-Focused Brief Therapy (SFBT):
   - Focus on solutions rather than problems
   - Use the miracle question when appropriate
   - Look for exceptions to problems
   - Set achievable goals

4. Mindfulness-Based Techniques:
   - Encourage present-moment awareness
   - Teach simple grounding exercises
   - Promote non-judgmental acceptance
   - Guide brief meditation exercises

CRITICAL GUIDELINES:
1. Maintain Conversation Context:
   - Remember and reference previous parts of the conversation
   - Build upon earlier insights and discussions
   - Notice patterns in user's responses
   - Track emotional progress

2. Response Quality:
   - Provide substantive, meaningful responses
   - Use therapeutic techniques appropriately
   - Maintain a consistent therapeutic approach
   - Demonstrate deep understanding of user's concerns

3. Engagement:
   - Ask relevant follow-up questions
   - Reflect user's emotions accurately
   - Validate experiences while promoting growth
   - Guide toward practical insights

4. Professional Boundaries:
   - Maintain appropriate therapeutic distance
   - Be clear about limitations as AI
   - Refer to professional help when needed
   - Focus on emotional support and coping strategies

SAFETY PROTOCOLS:
- NEVER give medical advice
- NEVER diagnose conditions
- If someone is in crisis, direct them to emergency services
- For serious mental health concerns, always recommend professional help
- Be clear about your limitations as an AI assistant

CONVERSATION STRUCTURE:
1. Listen and validate feelings
2. Reflect understanding
3. Explore underlying thoughts/beliefs
4. Offer appropriate therapeutic techniques
5. Encourage actionable steps
6. Maintain hope and optimism

Remember: You must maintain conversation context and provide high-quality, consistent therapeutic support throughout the entire session."""

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not api_key:
            return jsonify({
                'error': 'OpenAI API key is not configured',
                'status': 'error'
            }), 500

        data = request.json
        user_message = data.get('message')
        message_history = data.get('history', [])
        
        if not user_message:
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

        print(f"Sending request to OpenAI with message history length: {len(messages)}")
        
        # Create chat completion with context
        response = openai.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better context handling
            messages=messages,
            temperature=0.7,
            max_tokens=500  # Increased token limit for more detailed responses
        )

        # Extract the assistant's message
        ai_message = response.choices[0].message.content
        print(f"Received response from OpenAI: {ai_message[:100]}...")

        return jsonify({
            'message': ai_message,
            'status': 'success'
        })

    except openai.error.OpenAIError as e:
        error_message = str(e)
        print(f"OpenAI Error: {error_message}")
        
        if "exceeded your current quota" in error_message:
            error_message = "The OpenAI API key has exceeded its quota. Please check your billing details."
        elif "invalid api key" in error_message.lower():
            error_message = "The OpenAI API key is invalid. Please check your API key configuration."
        
        return jsonify({
            'error': error_message,
            'status': 'error'
        }), 500

    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
