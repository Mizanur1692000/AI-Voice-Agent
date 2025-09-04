import os
import time
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
import speech_recognition as sr
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from gtts import gTTS
import tempfile
import asyncio
import edge_tts
import pyttsx3
import re
import warnings
import threading
import pyaudio
import requests
import json
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class RealTimeVoiceAgent:
    def __init__(self):
        self.setup_apis()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.2  # Increased pause threshold
        
        self.is_listening = False
        self.is_speaking = False
        self.conversation_history = []
        self.setup_tts()
        
        # For interruption functionality
        self.interrupt_flag = False
        self.audio_thread = None
        self.stop_audio = False
        
        # PyAudio for interruptible audio playback
        self.p = pyaudio.PyAudio()
        
    def setup_apis(self):
        """Initialize API clients with keys from environment variables"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=google_api_key)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def setup_tts(self):
        """Set up text-to-speech engines"""
        self.pyttsx_engine = None
        try:
            self.pyttsx_engine = pyttsx3.init()
            voices = self.pyttsx_engine.getProperty('voices')
            if len(voices) > 0:
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower():
                        self.pyttsx_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.pyttsx_engine.setProperty('voice', voices[0].id)
            
            self.pyttsx_engine.setProperty('rate', 160)
            self.pyttsx_engine.setProperty('volume', 0.9)
        except Exception:
            self.pyttsx_engine = None
    
    def get_cricket_match_info(self, team1, team2, date=None):
        """Get cricket match information using a sports API"""
        try:
            # Using CricAPI (you would need to sign up for an API key)
            api_key = os.getenv("CRICAPI_KEY")
            if not api_key:
                return "Cricket API key not available. Please check the documentation."
            
            # Construct the API URL
            if date:
                url = f"https://cricapi.com/api/matches?apikey={api_key}&team={team1}&team={team2}&date={date}"
            else:
                url = f"https://cricapi.com/api/matches?apikey={api_key}&team={team1}&team={team2}"
            
            response = requests.get(url)
            data = response.json()
            
            if data.get("matches"):
                match = data["matches"][0]
                team1 = match["team-1"]
                team2 = match["team-2"]
                match_type = match["type"]
                match_date = match["date"]
                status = match["matchStarted"] 
                
                if status:
                    if "winner" in match:
                        winner = match["winner"]
                        return f"The match between {team1} and {team2} on {match_date} was won by {winner}."
                    else:
                        return f"The match between {team1} and {team2} on {match_date} is in progress."
                else:
                    return f"The match between {team1} and {team2} is scheduled for {match_date}."
            else:
                return "No recent matches found between these teams."
                
        except Exception as e:
            print(f"Cricket API error: {e}")
            return "I couldn't retrieve the cricket match information at the moment."
    
    def get_current_date(self):
        """Get current date in a readable format"""
        return datetime.now().strftime("%B %d, %Y")
    
    def generate_response(self, user_input):
        """Generate a response using Gemini with conversation history"""
        # Check for specific queries that need real-time information
        if any(word in user_input.lower() for word in ["cricket", "match", "score", "sports"]):
            if "bangladesh" in user_input.lower() and "netherland" in user_input.lower():
                # Extract date if mentioned
                date_match = re.search(r"\b(\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b", user_input, re.IGNORECASE)
                date = date_match.group(1) if date_match else None
                
                match_info = self.get_cricket_match_info("Bangladesh", "Netherlands", date)
                return match_info
        
        messages = []
        
        system_prompt = """You're a helpful, friendly AI assistant, always ready for a casual, natural conversation. 
        Keep responses clear, concise, and conversational, with a tone that feels like chatting with a real person.
        Do not use any emojis, markdown formatting, asterisks, bullets, or other special characters in your response.
        If you don't have real-time information, be honest about it."""
        
        messages.append(SystemMessage(content=system_prompt))
        
        for msg in self.conversation_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(SystemMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
        
        try:
            # Get response from Gemini
            response = self.llm.invoke(messages)
            
            # Clean up response for better TTS
            cleaned_response = self.clean_text_for_speech(response.content)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})
            
            return cleaned_response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def clean_text_for_speech(self, text):
        """Clean text to make it more natural for TTS"""
        # Remove all markdown formatting and special characters
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#+', '', text)
        
        # Remove bullet points and other special characters
        text = re.sub(r'[-•*]\s*', '', text)  # Remove bullet points
        text = re.sub(r'\[.*?\]', '', text)    # Remove anything in brackets
        text = re.sub(r'\(.*?\)', '', text)    # Remove anything in parentheses
        
        # Remove emojis and other non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Remove colons and other punctuation that might cause issues
        text = re.sub(r'[:;]', ',', text)
        
        # Replace common technical terms with more spoken forms
        replacements = {
            "AI": "A I", "API": "A P I", "URL": "U R L", 
            "HTML": "H T M L", "CSS": "C S S", "JSON": "Jason",
            "GPT": "G P T", "LLM": "L L M", "RAG": "R A G",
        }
        
        for term, replacement in replacements.items():
            text = text.replace(term, replacement)
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        return text
    
    def play_audio_interruptible(self, audio_segment):
        """Play audio with interrupt capability using PyAudio"""
        try:
            # Convert to raw data
            raw_data = audio_segment.raw_data
            sample_width = audio_segment.sample_width
            frame_rate = audio_segment.frame_rate
            channels = audio_segment.channels
            
            # Open stream
            stream = self.p.open(
                format=self.p.get_format_from_width(sample_width),
                channels=channels,
                rate=frame_rate,
                output=True
            )
            
            # Play audio in chunks
            chunk_size = 1024
            index = 0
            while index < len(raw_data) and not self.stop_audio:
                chunk = raw_data[index:index + chunk_size]
                stream.write(chunk)
                index += chunk_size
                time.sleep(0.01)  # Small delay to allow interruption check
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            
            return not self.stop_audio
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
    
    async def edge_tts_to_audio(self, text):
        """Convert text to audio using Edge TTS"""
        try:
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmpfile = f.name
            
            await communicate.save(tmpfile)
            
            # Load audio file
            audio_segment = AudioSegment.from_mp3(tmpfile)
            
            # Clean up
            os.unlink(tmpfile)
            
            return audio_segment
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return None
    
    def google_tts_to_audio(self, text):
        """Convert text to audio using Google TTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmpfile = f.name
            
            tts.save(tmpfile)
            
            # Load audio file
            audio_segment = AudioSegment.from_mp3(tmpfile)
            
            # Clean up
            os.unlink(tmpfile)
            
            return audio_segment
        except Exception as e:
            print(f"Google TTS error: {e}")
            return None
    
    def text_to_speech_thread(self, text):
        """Thread function for text-to-speech with interruption support"""
        self.is_speaking = True
        self.stop_audio = False
        self.interrupt_flag = False
        
        # Start interruption detection in a separate thread
        interrupt_thread = threading.Thread(target=self.listen_for_interruptions)
        interrupt_thread.daemon = True
        interrupt_thread.start()
        
        # Try Edge TTS first for most natural voice
        audio_segment = asyncio.run(self.edge_tts_to_audio(text))
        
        # If interrupted, don't try other methods
        if self.interrupt_flag:
            self.is_speaking = False
            return
            
        # Try Google TTS if Edge TTS failed
        if audio_segment is None:
            audio_segment = self.google_tts_to_audio(text)
            
        # If we have audio, play it with interruption support
        if audio_segment is not None:
            success = self.play_audio_interruptible(audio_segment)
            if not success:
                print("Audio playback was interrupted")
        
        self.is_speaking = False
    
    def text_to_speech(self, text):
        """Convert text to speech using the best available option"""
        if not text.strip():
            return
            
        # Start TTS in a separate thread
        self.audio_thread = threading.Thread(target=self.text_to_speech_thread, args=(text,))
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def listen_for_interruptions(self):
        """Listen for user interruptions while the AI is speaking"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 800  # Higher threshold to reduce false positives
        recognizer.dynamic_energy_threshold = False
        
        while self.is_speaking and not self.interrupt_flag:
            try:
                with sr.Microphone() as source:
                    # Use a shorter adjustment time
                    recognizer.adjust_for_ambient_noise(source, duration=0.1)
                    try:
                        # Listen for very short periods to quickly detect interruptions
                        audio = recognizer.listen(source, timeout=0.3, phrase_time_limit=0.3)
                        # If we get audio, set the interrupt flag
                        self.interrupt_flag = True
                        self.stop_audio = True
                        print("💬 Interruption detected! Stopping playback...")
                        break
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        continue
                    except Exception as e:
                        continue
            except Exception as e:
                continue
    
    def listen_for_speech(self):
        """Listen for speech using microphone"""
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                try:
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.WaitTimeoutError:
                    return None
                except sr.UnknownValueError:
                    return None
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    return None
                    
        except Exception as e:
            print(f"Microphone error: {e}")
            return None
    
    def process_conversation(self):
        """Main conversation processing loop"""
        print("🎙️  Voice conversation started. Let's chat!")
        print("💬 Say 'exit', 'quit', or 'stop' to end the conversation")
        print("🗣️  You can interrupt me anytime by starting to speak!")
        print("🏏 Cricket match information available for real-time queries")
        
        # Start with a welcome message
        print("🤖 AI: Hello there! What would you like to talk about?")
        
        while self.is_listening:
            # Wait if currently speaking
            if self.is_speaking:
                # Wait for speech to finish or be interrupted
                while self.is_speaking:
                    time.sleep(0.1)
                
                # If interrupted, process the new input immediately
                if self.interrupt_flag:
                    self.interrupt_flag = False
                    self.stop_audio = False
                    continue
                
            # Listen for speech
            text = self.listen_for_speech()
            
            if text and text.strip():
                print(f"👤 You: {text}")
                
                # Check for exit command
                if text.lower() in ['exit', 'quit', 'stop', 'goodbye', 'end']:
                    print("🤖 AI: Goodbye!")
                    break
                
                # Generate and speak response
                response = self.generate_response(text)
                print(f"🤖 AI: {response}")
                self.text_to_speech(response)
    
    def start_conversation(self):
        """Start the voice conversation"""
        self.is_listening = True
        
        try:
            self.process_conversation()
        except KeyboardInterrupt:
            print("\nStopping voice agent...")
        finally:
            self.is_speaking = False
            self.is_listening = False
            self.stop_audio = True
            
            # Clean up PyAudio
            self.p.terminate()

if __name__ == "__main__":
    agent = RealTimeVoiceAgent()
    agent.start_conversation()