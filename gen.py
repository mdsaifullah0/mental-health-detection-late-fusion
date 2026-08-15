from groq import Groq
import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MentalHealthLLM:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mental Health LLM with Groq client
        
        Args:
            api_key: Groq API key. If None, will use GROQ_API_KEY environment variable
        """
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  
        
        # Context templates for different mental health conditions
        self.context_templates = {
            'Depression': {
                'system_prompt': """You are a compassionate mental health support assistant. The user has been classified as showing signs of depression in their text. 
                Provide empathetic, supportive responses while encouraging professional help. Focus on:
                - Validation of their feelings
                - Gentle encouragement
                - Self-care suggestions
                - Professional resources
                - Hope and recovery-oriented messaging
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "I understand you may be going through a difficult time. Depression can feel overwhelming, but please know that you're not alone and that help is available. Would you like to talk about what you're experiencing, or would you prefer some suggestions for coping strategies?"
            },
            'Anxiety': {
                'system_prompt': """You are a supportive mental health assistant. The user has been classified as showing signs of anxiety in their text.
                Provide calming, reassuring responses while offering practical anxiety management techniques. Focus on:
                - Grounding techniques
                - Breathing exercises
                - Cognitive reframing
                - Stress management
                - Encouraging professional support
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "I can sense you might be feeling anxious or worried. Anxiety can be really challenging, but there are many effective ways to manage it. Would you like to try some breathing exercises together, or would you prefer to talk about what's causing your anxiety?"
            },
            'Stress': {
                'system_prompt': """You are a helpful mental health support assistant. The user has been classified as showing signs of stress in their text.
                Provide practical stress management advice and emotional support. Focus on:
                - Stress reduction techniques
                - Time management
                - Relaxation methods
                - Healthy coping strategies
                - Work-life balance
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "It sounds like you're dealing with stress right now. Stress is a normal part of life, but it's important to manage it effectively. Would you like to explore some stress-reduction techniques, or would you prefer to talk about what's causing your stress?"
            },
            'Suicidal': {
                'system_prompt': """You are a crisis-aware mental health assistant. The user has been classified as showing signs of suicidal ideation in their text.
                This is a CRITICAL situation. Provide immediate crisis resources and empathetic support. Focus on:
                - Immediate safety
                - Crisis hotlines and resources
                - Encouraging immediate professional help
                - Validation without judgment
                - Hope and connection
                
                ALWAYS provide crisis resources and encourage immediate professional help.""",
                'initial_response': "I'm really concerned about you and want to help. If you're having thoughts of hurting yourself, please reach out for immediate support:\n\n🚨 Crisis Resources:\n• National Suicide Prevention Lifeline: 988 (US)\n• Crisis Text Line: Text HOME to 741741\n• Emergency Services: 911\n\nYou matter, and there are people who want to help you through this difficult time. Can you tell me if you're in a safe place right now?"
            },
            'Bipolar': {
                'system_prompt': """You are a knowledgeable mental health support assistant. The user has been classified as showing signs consistent with bipolar patterns in their text.
                Provide supportive, balanced responses about mood management. Focus on:
                - Mood tracking and awareness
                - Routine and stability
                - Professional treatment importance
                - Medication compliance (if applicable)
                - Lifestyle factors
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "I notice there might be some mood-related patterns in what you've shared. Managing mood changes can be challenging, but with the right support and strategies, it's very manageable. Would you like to discuss mood management techniques, or is there something specific you'd like to talk about?"
            },
            'Personality disorder': {
                'system_prompt': """You are a non-judgmental mental health support assistant. The user has been classified as showing patterns that might relate to personality concerns.
                Provide supportive, validating responses while focusing on healthy coping strategies. Focus on:
                - Emotional regulation
                - Relationship skills
                - Self-awareness
                - Therapeutic approaches
                - Reducing stigma
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "I understand that navigating emotions and relationships can sometimes feel challenging. Everyone has unique ways of experiencing and responding to the world. Would you like to explore some strategies for emotional well-being, or is there something specific on your mind?"
            },
            'Normal': {
                'system_prompt': """You are a supportive mental health and wellness assistant. The user's text appears to be within normal ranges.
                Provide general mental health support and wellness advice. Focus on:
                - Mental health maintenance
                - Stress prevention
                - Healthy lifestyle choices
                - Building resilience
                - General wellness tips
                
                Always remind users that you're not a replacement for professional mental health care.""",
                'initial_response': "It's great that you're taking time to check in with your mental health! Maintaining good mental wellness is important for everyone. Is there anything specific about your mental health or well-being that you'd like to discuss or learn more about?"
            }
        }
        
        # General crisis keywords to watch for
        self.crisis_keywords = [
            'kill myself', 'suicide', 'end it all', 'hurt myself', 'self harm',
            'cutting', 'overdose', 'jump', 'hang', 'gun', 'pills', 'die',
            'better off dead', 'no point', 'give up', 'cant go on'
        ]
    
    def get_initial_response(self, predicted_class: str, confidence: float) -> str:
        """
        Get initial response based on the predicted mental health class
        
        Args:
            predicted_class: The predicted mental health category
            confidence: Confidence score of the prediction
            
        Returns:
            Initial response string
        """
        context = self.context_templates.get(predicted_class, self.context_templates['Normal'])
        
        # Add confidence information if it's low
        response = context['initial_response']
        if confidence < 0.6:
            response += f"\n\nNote: The AI classification had moderate confidence ({confidence*100:.1f}%), so please feel free to share more if this doesn't seem to match your situation."
        
        return response
    
    def check_crisis_content(self, message: str) -> bool:
        """
        Check if the message contains crisis-related content
        
        Args:
            message: User's message
            
        Returns:
            True if crisis keywords are detected
        """
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.crisis_keywords)
    
    def generate_response(self, message: str, predicted_class: str, chat_history: List[Dict] = None) -> str:
        """
        Generate a response using the LLM
        
        Args:
            message: User's message
            predicted_class: The predicted mental health category
            chat_history: Previous conversation history
            
        Returns:
            Generated response string
        """
        try:
            # Check for crisis content
            if self.check_crisis_content(message):
                return self._get_crisis_response()
            
            # Get context for the predicted class
            context = self.context_templates.get(predicted_class, self.context_templates['Normal'])
            
            # Build messages for the API
            messages = [
                {
                    "role": "system",
                    "content": context['system_prompt']
                }
            ]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history)
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Generate response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=False,
                stop=None,
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"LLM Error: {str(e)}")  # Log the error
            return f"I'm sorry, I'm having trouble generating a response right now. Please try again later. If you're in crisis, please contact emergency services or a crisis helpline immediately."
    
    def generate_response_stream(self, message: str, predicted_class: str, chat_history: List[Dict] = None):
        """
        Generate a streaming response using the LLM
        
        Args:
            message: User's message
            predicted_class: The predicted mental health category
            chat_history: Previous conversation history
            
        Yields:
            Streaming response chunks
        """
        try:
            # Check for crisis content
            if self.check_crisis_content(message):
                yield self._get_crisis_response()
                return
            
            # Get context for the predicted class
            context = self.context_templates.get(predicted_class, self.context_templates['Normal'])
            
            # Build messages for the API
            messages = [
                {
                    "role": "system",
                    "content": context['system_prompt']
                }
            ]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history)
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Generate streaming response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=True,
                stop=None,
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"LLM Stream Error: {str(e)}")  # Log the error
            yield f"I'm sorry, I'm having trouble generating a response right now. Please try again later. If you're in crisis, please contact emergency services or a crisis helpline immediately."
    
    def _get_crisis_response(self) -> str:
        """
        Get immediate crisis response
        
        Returns:
            Crisis response with resources
        """
        return """🚨 IMMEDIATE CRISIS SUPPORT NEEDED 🚨

I'm very concerned about what you've shared. Please reach out for immediate help:

📞 CRISIS RESOURCES:
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
• Emergency Services: 911 (US), 999 (UK), 112 (EU)

🏥 If you're in immediate danger, please:
• Call emergency services (911)
• Go to your nearest emergency room
• Call a trusted friend or family member
• Contact your mental health provider

You are not alone, and your life has value. There are people who want to help you through this difficult time. These feelings can change, and support is available.

Please stay safe and reach out for help immediately."""

    def get_mental_health_resources(self, predicted_class: str) -> Dict:
        """
        Get relevant mental health resources based on predicted class
        
        Args:
            predicted_class: The predicted mental health category
            
        Returns:
            Dictionary of resources
        """
        resources = {
            'general': {
                'name': 'General Mental Health Resources',
                'resources': [
                    'National Alliance on Mental Illness (NAMI): nami.org',
                    'Mental Health America: mhanational.org',
                    'Psychology Today Therapist Finder: psychologytoday.com',
                    'Crisis Text Line: Text HOME to 741741'
                ]
            },
            'Depression': {
                'name': 'Depression Resources',
                'resources': [
                    'Depression and Bipolar Support Alliance: dbsalliance.org',
                    'National Institute of Mental Health: nimh.nih.gov/health/topics/depression',
                    'Befrienders Worldwide: befrienders.org',
                    'International Association for Suicide Prevention: iasp.info'
                ]
            },
            'Anxiety': {
                'name': 'Anxiety Resources',
                'resources': [
                    'Anxiety and Depression Association of America: adaa.org',
                    'National Institute of Mental Health: nimh.nih.gov/health/topics/anxiety-disorders',
                    'Calm App: calm.com',
                    'Headspace: headspace.com'
                ]
            },
            'Stress': {
                'name': 'Stress Management Resources',
                'resources': [
                    'American Psychological Association: apa.org/topics/stress',
                    'Mayo Clinic Stress Management: mayoclinic.org/healthy-lifestyle/stress-management',
                    'Mindfulness-Based Stress Reduction: palousemindfulness.com',
                    'Workplace Stress Resources: cdc.gov/niosh/topics/stress'
                ]
            },
            'Suicidal': {
                'name': 'Crisis Resources',
                'resources': [
                    'National Suicide Prevention Lifeline: 988',
                    'Crisis Text Line: Text HOME to 741741',
                    'International Association for Suicide Prevention: iasp.info',
                    'American Foundation for Suicide Prevention: afsp.org',
                    'The Trevor Project (LGBTQ+): thetrevorproject.org'
                ]
            },
            'Bipolar': {
                'name': 'Bipolar Disorder Resources',
                'resources': [
                    'Depression and Bipolar Support Alliance: dbsalliance.org',
                    'National Institute of Mental Health: nimh.nih.gov/health/topics/bipolar-disorder',
                    'International Bipolar Foundation: ibpf.org',
                    'Bipolar Disorder Association: bdassociation.org'
                ]
            },
            'Personality disorder': {
                'name': 'Personality Disorder Resources',
                'resources': [
                    'National Alliance on Mental Illness: nami.org/About-Mental-Illness/Mental-Health-Conditions/Personality-Disorders',
                    'Borderline Personality Disorder Resource Center: borderlinepersonalitydisorder.org',
                    'Psychology Today: psychologytoday.com',
                    'National Education Alliance for Borderline Personality Disorder: borderlinepersonalitydisorder.org'
                ]
            }
        }
        
        return resources.get(predicted_class, resources['general'])