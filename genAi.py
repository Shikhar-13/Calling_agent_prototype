import torch
from transformers import pipeline
import whisper
import pyaudio
import wave
import numpy as np
import time
import json
import os
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from gtts import gTTS
import tempfile
import playsound
from pyaudio import paInt16
import traceback

class CallAnalytics:
    def __init__(self):
        self.call_duration: float = 0
        self.customer_talk_time: float = 0
        self.agent_talk_time: float = 0
        self.turn_count: int = 0
        self.positive_responses: int = 0
        self.negative_responses: int = 0
        self.sentiment_scores: List[float] = []
        
    def to_dict(self) -> Dict:
        return {
            "call_duration": self.call_duration,
            "customer_talk_time": self.customer_talk_time,
            "agent_talk_time": self.agent_talk_time,
            "turn_count": self.turn_count,
            "positive_responses": self.positive_responses,
            "negative_responses": self.negative_responses,
            "average_sentiment": sum(self.sentiment_scores) / len(self.sentiment_scores) if self.sentiment_scores else 0
        }

class AIColdCaller:
    def __init__(
        self,
        llm_model: str = "facebook/opt-350m",
        log_dir: str = "call_logs",
        system_prompt: str = None
    ):
        # Initialize logging first to capture any initialization errors
        self.setup_logging(log_dir)
        
        try:
            # Initialize models
            self.setup_models(llm_model)
            
            # Initialize analytics
            self.analytics = CallAnalytics()
            
            # Set up system prompt
            self.system_prompt = system_prompt or self.get_default_system_prompt()
            
            self.conversation_history = []
            self.call_start_time = None
            
            # Verify audio device availability
            
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
            raise
    def setup_logging(self, log_dir: str) -> None:
        """Set up logging directory and configuration"""
        try:
            # Store log directory path
            self.log_dir = log_dir
            
            # Create necessary directories
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, "transcripts"), exist_ok=True)
            os.makedirs(os.path.join(log_dir, "analytics"), exist_ok=True)
            os.makedirs(os.path.join(log_dir, "recordings"), exist_ok=True)
            
            # Set up logging configuration
            log_file = os.path.join(log_dir, "cold_caller.log")
            
            # Create a formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Set up file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            
            # Set up console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            
            # Remove any existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            # Add our handlers
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            logging.info(f"Logging initialized. Log directory: {log_dir}")
            
        except Exception as e:
            # If we fail to set up logging, print to console as fallback
            print(f"Failed to initialize logging: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
    def setup_models(self, llm_model: str) -> None:
        """Initialize all AI models and services"""
        try:
            logging.info("Initializing AI models...")
            
            # Initialize Whisper
            self.whisper_model = whisper.load_model("base",device='cpu')
            logging.info("Whisper model initialized")
            
            # Initialize language model
            self.conversation = pipeline(
                "text-generation",
                model=llm_model,
                torch_dtype=torch.float32,
                device_map="auto",
                model_kwargs={"temperature": 0.7}
            )
            logging.info("Language model initialized")
            
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logging.info("Sentiment analyzer initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize models: {str(e)}")
            raise
    def get_default_system_prompt(self) -> str:
        """Return the default system prompt with sales script"""
        return """You are a professional sales agent for a software company. Your goal is to schedule a demo of our 
        project management software. Follow this script structure:

        1. Introduction:
        - Introduce yourself and the company
        - Ask if you've caught them at a good time
        
        2. Qualification:
        - Ask about their current project management process
        - Identify pain points
        
        3. Value Proposition:
        - Explain how our software addresses their specific challenges
        - Share relevant case studies or success stories
        
        4. Close:
        - Suggest scheduling a demo
        - Provide flexible timing options
        
        Guidelines:
        - Keep responses concise and natural
        - Show empathy and active listening
        - Respect rejection professionally
        - Use positive language
        - Address objections with understanding
        
        Key Features to Highlight:
        - Task automation
        - Real-time collaboration
        - Custom workflows
        - Integration capabilities
        - 24/7 support
        
        Demo Booking Options:
        - 30-minute quick demo
        - 60-minute comprehensive walkthrough
        - Custom team presentation"""

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text and return score"""
        result = self.sentiment_analyzer(text)[0]
        score = 1.0 if result["label"] == "POSITIVE" else 0.0
        self.analytics.sentiment_scores.append(score)
        return score

    def detect_response_type(self, text: str) -> str:
        """Detect if response is positive, negative, or neutral"""
        positive_keywords = ["yes", "sure", "interested", "good", "okay", "great"]
        negative_keywords = ["no", "not", "busy", "later", "don't"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in positive_keywords):
            self.analytics.positive_responses += 1
            return "positive"
        elif any(word in text_lower for word in negative_keywords):
            self.analytics.negative_responses += 1
            return "negative"
        return "neutral"

    def record_audio(self, duration: int = 5) -> str:
        """Record audio from microphone and return the filename"""
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        start_time = time.time()
        
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        recording_duration = time.time() - start_time
        self.analytics.customer_talk_time += recording_duration
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/recordings/recording_{timestamp}.wav"
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Log the filename to confirm it was created
        logging.info(f"Audio recorded and saved to: {filename}")
        if not os.path.exists(os.path.dirname(filename)):
             os.makedirs(os.path.dirname(filename))
        
        return filename
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper"""
        try:
            logging.info(f"Transcribing audio file: {audio_file}")
            result = self.whisper_model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Transcription error: {str(e)} for file: {audio_file}")
            return ""

    def generate_response(self, user_input: str) -> str:
        """Generate response using language model"""
        try:
            # Analyze customer input
            sentiment = self.analyze_sentiment(user_input)
            response_type = self.detect_response_type(user_input)
            
            # Construct prompt with conversation history and analytics
            prompt = f"{self.system_prompt}\n\nConversation history:\n"
            for turn in self.conversation_history[-3:]:
                prompt += f"{turn}\n"
            
            prompt += f"\nCustomer sentiment: {'positive' if sentiment > 0.5 else 'negative'}"
            prompt += f"\nResponse type: {response_type}"
            prompt += f"\nCustomer: {user_input}\nAgent:"
            
            # Generate response with adjusted parameters
            response = self.conversation(
                prompt,
                max_length=400,
                num_return_sequences=1,
                top_p=0.9,
                do_sample=True,
                no_repeat_ngram_size=2,
                truncation=True 
            )[0]["generated_text"]
            
            # Extract only the agent's response
            response = response.split("Agent:")[-1].strip()
            
            # Clean up response if needed
            if not response or response.isspace():
                response = "I understand. Could you please tell me more about your needs?"
            
            # Update conversation history
            self.conversation_history.append(f"Customer: {user_input}")
            self.conversation_history.append(f"Agent: {response}")
            
            self.analytics.turn_count += 1
            
            return response
            
        except Exception as e:
            logging.error(f"Response generation error: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. Could you please repeat that?"

    def speak_response(self, text: str) -> None:
        """Convert text to speech using gTTS"""
        try:
            # Use gTTS to synthesize speech
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                tts.save(temp_audio.name)
                temp_audio_path = temp_audio.name
            
            # Play the generated audio
            playsound.playsound(temp_audio_path)
            os.remove(temp_audio_path)

            # Update analytics
            self.analytics.agent_talk_time += len(text.split()) / 2  # Approximation
            logging.info("Speech synthesis completed")
        except Exception as e:
            logging.error(f"Speech synthesis error: {str(e)}")

    def save_conversation(self) -> None:
        """Save conversation transcript and analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save transcript
        transcript_file = f"{self.log_dir}/transcripts/transcript_{timestamp}.txt"
        with open(transcript_file, 'w') as f:
            f.write("\n".join(self.conversation_history))
        
        # Save analytics
        analytics_file = f"{self.log_dir}/analytics/analytics_{timestamp}.json"
        with open(analytics_file, 'w') as f:
            json.dump(self.analytics.to_dict(), f, indent=4)
        
        # Update analytics summary
        self.update_analytics_summary()
        
    def update_analytics_summary(self) -> None:
        """Update the running summary of all calls"""
        summary_file = f"{self.log_dir}/analytics/summary.csv"
        
        current_analytics = pd.DataFrame([self.analytics.to_dict()])
        current_analytics['timestamp'] = datetime.now()
        
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            summary_df = pd.concat([summary_df, current_analytics])
        else:
            summary_df = current_analytics
        
        summary_df.to_csv(summary_file, index=False)

    def run_conversation(self) -> None:
        """Run the main conversation loop"""
        print("AI Cold Caller started. Press Ctrl+C to exit.")
        logging.info("Starting new call")
        
        self.call_start_time = time.time()
        
        try:
            while True:
                # Record audio
                print("\nListening...")
                audio_file = self.record_audio()
                
                # Transcribe audio
                user_input = self.transcribe_audio(audio_file)
                print(f"\nCustomer: {user_input}")
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"\nAgent: {response}")
                
                # Speak response
                self.speak_response(response)
                
        except KeyboardInterrupt:
            print("\nEnding conversation...")
            self.analytics.call_duration = time.time() - self.call_start_time
            self.save_conversation()
            logging.info("Call ended")
            
if __name__ == "__main__":
    # Initialize the AI Cold Caller
    agent = AIColdCaller()
    
    # Run the conversation
    agent.run_conversation()