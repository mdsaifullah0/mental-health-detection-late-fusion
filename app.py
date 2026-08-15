from flask import Flask, request, render_template, jsonify, session
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertModel, XLMRobertaModel
import numpy as np
import re
from datetime import datetime
import uuid
from gen import MentalHealthLLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Define your Late Fusion model architecture (matching your training code exactly)
class HybridTransformerLateFusion(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridTransformerLateFusion, self).__init__()
        # DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # XLM-RoBERTa model
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        # Freeze the base models (matching your training setup)
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        for param in self.xlmr.parameters():
            param.requires_grad = False
            
        # Unfreeze the last layers (fine-tuning)
        for param in self.distilbert.transformer.layer[-2:].parameters():
            param.requires_grad = True
            
        for param in self.xlmr.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Feature dimensions
        self.distilbert_dim = 768
        self.xlmr_dim = 768
        
        # Individual classifiers (matching your exact architecture)
        self.dropout = nn.Dropout(0.3)
        self.distilbert_classifier = nn.Sequential(
            nn.Linear(self.distilbert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.xlmr_classifier = nn.Sequential(
            nn.Linear(self.xlmr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Weights for ensemble (learnable)
        self.model_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, input_ids1, attention_mask1, token_type_ids1, 
                input_ids2, attention_mask2):
        # Get embeddings from DistilBERT
        distilbert_output = self.distilbert(
            input_ids=input_ids1,
            attention_mask=attention_mask1
        )
        distilbert_embeddings = distilbert_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Get embeddings from XLM-RoBERTa
        xlmr_output = self.xlmr(
            input_ids=input_ids2,
            attention_mask=attention_mask2
        )
        xlmr_embeddings = xlmr_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Get individual model predictions
        distilbert_logits = self.distilbert_classifier(self.dropout(distilbert_embeddings))
        xlmr_logits = self.xlmr_classifier(self.dropout(xlmr_embeddings))
        
        # Normalize weights
        weights = F.softmax(self.model_weights, dim=0)
        
        # Late fusion: weighted average of logits
        combined_logits = weights[0] * distilbert_logits + weights[1] * xlmr_logits
        
        return combined_logits, weights, distilbert_logits, xlmr_logits

# Class names mapping
CLASS_NAMES = [
    'Anxiety',
    'Bipolar',
    'Depression',
    'Normal',
    'Personality disorder',
    'Stress',
    'Suicidal'
]

# Global variables for model and tokenizers
model = None
distilbert_tokenizer = None
xlmr_tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize LLM with proper error handling
def initialize_llm():
    """Initialize the LLM with proper API key validation"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("   Please set your Groq API key in the .env file")
        print("   You can get an API key from: https://console.groq.com/keys")
        return None
    
    if len(groq_api_key.strip()) < 20:  # Basic validation
        print("❌ GROQ_API_KEY appears to be invalid (too short)")
        print("   Please check your API key in the .env file")
        return None
    
    try:
        llm = MentalHealthLLM(api_key=groq_api_key)
        print("✅ LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {str(e)}")
        return None

# Initialize LLM
llm = initialize_llm()

# Store chat sessions (in production, use a database)
chat_sessions = {}

def load_model():
    """Load the trained late fusion model and tokenizers"""
    global model, distilbert_tokenizer, xlmr_tokenizer
    
    try:
        # Load tokenizers
        distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        xlmr_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        
        # Initialize model architecture
        model = HybridTransformerLateFusion(num_classes=len(CLASS_NAMES))
        
        # Load the saved model - handle different saving formats
        try:
            # Method 1: Try loading as state_dict directly
            checkpoint = torch.load('fusion_model.pt', map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)
            print("✅ Loaded model using direct state_dict method")
            
        except Exception as e1:
            try:
                # Method 2: Try loading as full model (weights_only=False)
                checkpoint = torch.load('fusion_model.pt', map_location=device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    # Handle different dictionary structures
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Loaded model using 'model_state_dict' key")
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                        print("✅ Loaded model using 'state_dict' key")
                    else:
                        # Try using the checkpoint as state_dict
                        model.load_state_dict(checkpoint)
                        print("✅ Loaded model using checkpoint as state_dict")
                else:
                    # If it's not a dictionary, it might be the model itself
                    model = checkpoint
                    print("✅ Loaded complete model object")
                    
            except Exception as e2:
                # Method 3: Try loading without weights_only restriction
                try:
                    checkpoint = torch.load('fusion_model.pt', map_location=device)
                    
                    if hasattr(checkpoint, 'state_dict'):
                        model.load_state_dict(checkpoint.state_dict())
                        print("✅ Loaded model using checkpoint.state_dict()")
                    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Loaded model using checkpoint['model_state_dict']")
                    else:
                        model.load_state_dict(checkpoint)
                        print("✅ Loaded model using checkpoint as state_dict")
                        
                except Exception as e3:
                    print(f"❌ All loading methods failed:")
                    print(f"   Method 1 error: {str(e1)}")
                    print(f"   Method 2 error: {str(e2)}")
                    print(f"   Method 3 error: {str(e3)}")
                    return False
        
        model.to(device)
        model.eval()
        
        print(f"✅ Late Fusion model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_text(text):
    """Clean and preprocess input text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
    
    return text

def predict_mental_health(text, max_length=128):
    """Make prediction on input text using late fusion model"""
    if model is None or distilbert_tokenizer is None or xlmr_tokenizer is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize for DistilBERT
        distilbert_encoded = distilbert_tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Tokenize for XLM-RoBERTa
        xlmr_encoded = xlmr_tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        distilbert_input_ids = distilbert_encoded['input_ids'].to(device)
        distilbert_attention_mask = distilbert_encoded['attention_mask'].to(device)
        distilbert_token_type_ids = distilbert_encoded.get('token_type_ids', torch.zeros((1, max_length), device=device)).to(device)
        xlmr_input_ids = xlmr_encoded['input_ids'].to(device)
        xlmr_attention_mask = xlmr_encoded['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs, weights, distilbert_logits, xlmr_logits = model(
                distilbert_input_ids, distilbert_attention_mask, distilbert_token_type_ids,
                xlmr_input_ids, xlmr_attention_mask
            )
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Individual model predictions
            distilbert_probs = torch.softmax(distilbert_logits, dim=1)
            xlmr_probs = torch.softmax(xlmr_logits, dim=1)
            
            distilbert_pred = torch.argmax(distilbert_probs, dim=1).item()
            xlmr_pred = torch.argmax(xlmr_probs, dim=1).item()
        
        # Get all class probabilities for detailed results
        all_probs = probabilities[0].cpu().numpy()
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                'class': class_name,
                'probability': float(all_probs[i]),
                'percentage': f"{all_probs[i]*100:.2f}%"
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.2f}%",
            'all_predictions': results,
            'model_details': {
                'fusion_weights': weights.cpu().numpy().tolist(),
                'distilbert_prediction': CLASS_NAMES[distilbert_pred],
                'xlmr_prediction': CLASS_NAMES[xlmr_pred],
                'distilbert_confidence': f"{distilbert_probs[0][distilbert_pred].item()*100:.2f}%",
                'xlmr_confidence': f"{xlmr_probs[0][xlmr_pred].item()*100:.2f}%"
            }
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input text
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'Please provide some text for analysis'
            }), 400
        
        # Make prediction
        result, error = predict_mental_health(text)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Add metadata
        result['input_text'] = text
        result['timestamp'] = datetime.now().isoformat()
        result['text_length'] = len(text)
        
        # Create chat session and get initial response if LLM is available
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'chat_history': [],
            'created_at': datetime.now().isoformat()
        }
        
        result['chat_session_id'] = session_id
        
        # Get initial LLM response if available
        if llm:
            try:
                initial_response = llm.get_initial_response(
                    result['predicted_class'], 
                    result['confidence']
                )
                result['initial_llm_response'] = initial_response
            except Exception as e:
                result['initial_llm_response'] = f"LLM service temporarily unavailable. Classification results are still available above. Error: {str(e)}"
        else:
            result['initial_llm_response'] = "LLM service not available. Please check your API key configuration."
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with the LLM"""
    try:
        if not llm:
            return jsonify({'error': 'LLM service not available. Please check your API key configuration.'}), 503
        
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid or expired chat session'}), 400
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get chat session
        chat_session = chat_sessions[session_id]
        
        # Add user message to history
        chat_session['chat_history'].append({
            'role': 'user',
            'content': message
        })
        
        # Generate LLM response
        response = llm.generate_response(
            message,
            chat_session['predicted_class'],
            chat_session['chat_history'][:-1]  # Don't include the current message
        )
        
        # Add assistant response to history
        chat_session['chat_history'].append({
            'role': 'assistant',
            'content': response
        })
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Handle streaming chat requests with the LLM"""
    try:
        if not llm:
            return jsonify({'error': 'LLM service not available. Please check your API key configuration.'}), 503
        
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid or expired chat session'}), 400
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get chat session
        chat_session = chat_sessions[session_id]
        
        # Add user message to history
        chat_session['chat_history'].append({
            'role': 'user',
            'content': message
        })
        
        def generate():
            full_response = ""
            for chunk in llm.generate_response_stream(
                message,
                chat_session['predicted_class'],
                chat_session['chat_history'][:-1]
            ):
                full_response += chunk
                yield f"data: {chunk}\n\n"
            
            # Add complete response to history
            chat_session['chat_history'].append({
                'role': 'assistant',
                'content': full_response
            })
            
            yield "data: [DONE]\n\n"
        
        return app.response_class(generate(), mimetype='text/plain')
        
    except Exception as e:
        return jsonify({'error': f'Chat stream error: {str(e)}'}), 500

@app.route('/resources/<predicted_class>')
def get_resources(predicted_class):
    """Get mental health resources for a specific class"""
    try:
        if not llm:
            # Fallback resources if LLM is not available
            fallback_resources = {
                'resources': [
                    'National Alliance on Mental Illness (NAMI): nami.org',
                    'Crisis Text Line: Text HOME to 741741',
                    'National Suicide Prevention Lifeline: 988',
                    'Psychology Today: psychologytoday.com'
                ]
            }
            return jsonify(fallback_resources)
        
        resources = llm.get_mental_health_resources(predicted_class)
        return jsonify(resources)
    except Exception as e:
        return jsonify({'error': f'Error fetching resources: {str(e)}'}), 500

@app.route('/chat/history/<session_id>')
def get_chat_history(session_id):
    """Get chat history for a session"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'session_id': session_id,
            'chat_history': chat_sessions[session_id]['chat_history'],
            'predicted_class': chat_sessions[session_id]['predicted_class'],
            'confidence': chat_sessions[session_id]['confidence']
        })
        
    except Exception as e:
        return jsonify({'error': f'Error fetching chat history: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    llm_status = "available" if llm is not None else "not available"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'llm_status': llm_status,
        'model_type': 'Late Fusion (DistilBERT + XLM-RoBERTa)',
        'device': str(device),
        'classes': CLASS_NAMES,
        'llm_model': llm.model if llm else None,
        'active_sessions': len(chat_sessions)
    })

if __name__ == '__main__':
    print("Starting Mental Health Classification Server with LLM Support...")
    print("Loading Late Fusion model (DistilBERT + XLM-RoBERTa)...")
    
    # Check for GROQ API key
    if not os.environ.get('GROQ_API_KEY'):
        print("⚠️  Warning: GROQ_API_KEY environment variable not set.")
        print("   LLM chat features will not be available.")
        print("   Get your API key from: https://console.groq.com/keys")
    
    if load_model():
        print("✅ Late Fusion model loaded successfully!")
        print(f"Available classes: {', '.join(CLASS_NAMES)}")
        print("Model architecture: Late Fusion with learnable weights")
        if llm:
            print(f"LLM Model: {llm.model}")
        else:
            print("LLM: Not available")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure 'fusion_model.pt' exists in the same directory")
        print("2. Check that the model was saved from your training script")
        print("3. Set GROQ_API_KEY environment variable for LLM features")