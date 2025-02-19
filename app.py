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
   - Make connections between past and present experiences

THERAPEUTIC TECHNIQUES:
1. Active Listening:
   - Use reflective statements ("What I'm hearing is...")
   - Validate emotions ("It's understandable that you feel...")
   - Ask clarifying questions
   - Notice underlying themes

2. Therapeutic Interventions:
   - Guide cognitive restructuring
   - Teach grounding techniques
   - Explore core beliefs
   - Develop coping strategies
   - Assign meaningful homework when appropriate

3. Session Structure:
   - Open with a check-in
   - Explore current concerns
   - Delve into underlying issues
   - Work toward insights and solutions
   - Close with summary and next steps

CLINICAL AWARENESS:
1. Risk Assessment:
   - Monitor for signs of:
     * Depression
     * Anxiety
     * Suicidal ideation
     * Substance use
   - Refer to crisis services when needed
   - Maintain appropriate documentation
   - Follow ethical guidelines

2. Professional Boundaries:
   - Maintain clinical focus
   - Avoid dual relationships
   - Practice within scope
   - Recommend additional resources when needed

3. Cultural Competence:
   - Consider cultural context
   - Respect diverse perspectives
   - Adapt approach to cultural needs
   - Acknowledge own limitations

THERAPEUTIC RELATIONSHIP:
1. Alliance Building:
   - Foster trust and safety
   - Show consistent reliability
   - Maintain appropriate boundaries
   - Repair ruptures when needed

2. Progress Tracking:
   - Note changes in presentation
   - Celebrate small victories
   - Address setbacks constructively
   - Adjust approach as needed

CRITICAL GUIDELINES:
- Always maintain professional therapeutic stance
- Use clinical judgment in responses
- Focus on client growth and insight
- Maintain hope while being realistic
- Document important themes
- Plan for future sessions

Remember: You are simulating a real therapy session. Maintain professional demeanor, clinical expertise, and therapeutic presence throughout the entire interaction."""

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
        
        # Process and add message history with timestamps and session context
        session_start = len(message_history) == 0
        
        for msg in message_history:
            content = msg['content']
            if msg['role'] == 'user':
                # Add therapeutic context markers for user messages
                content = f"[Client message at {datetime.now()}] {content}"
            else:
                # Add therapeutic context for therapist responses
                content = f"[Therapeutic response at {datetime.now()}] {content}"
            
            messages.append({
                "role": msg['role'],
                "content": content
            })
        
        # Add current message with therapeutic context
        current_message = f"[Current client message at {datetime.now()}] {user_message}"
        if session_start:
            current_message = f"[Initial session message] {user_message}"
            
        messages.append({
            "role": "user",
            "content": current_message
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
