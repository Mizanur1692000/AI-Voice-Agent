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
from playsound import playsound
import asyncio
import edge_tts
import pyttsx3
import re
import warnings
import threading

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
        self.recognizer.pause_threshold = 0.8
        
        self.is_listening = False
        self.is_speaking = False
        self.conversation_history = []
        self.setup_tts()
        
        # For barge-in functionality
        self.interrupt_flag = False
        self.current_tts_process = None
        
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
    
    def generate_response(self, user_input):
        """Generate a response using Gemini with conversation history"""
        messages = []
        
        system_prompt = """You are a helpful, friendly AI assistant having a natural conversation. 
        Keep responses conversational and concise (1-2 sentences). 
        Respond like a real person would in a casual conversation."""
        
        messages.append(SystemMessage(content=system_prompt))
        
        for msg in self.conversation_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(SystemMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
        
        try:
            response = self.llm.invoke(messages)
            cleaned_response = self.clean_text_for_speech(response.content)
            
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})
            
            return cleaned_response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def clean_text_for_speech(self, text):
        """Clean text to make it more natural for TTS"""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#+', '', text)
        
        replacements = {
            "AI": "A I", "API": "A P I", "URL": "U R L", 
            "HTML": "H T M L", "CSS": "C S S", "JSON": "Jason",
            "GPT": "G P T", "LLM": "L L M", "RAG": "R A G",
        }
        
        for term, replacement in replacements.items():
            text = text.replace(term, replacement)
        
        return ' '.join(text.split())
    
    async def edge_tts_speech(self, text):
        """Use Edge TTS for high quality speech"""
        try:
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmpfile = f.name
            
            await communicate.save(tmpfile)
            
            # Check for interruptions before playing
            if self.interrupt_flag:
                os.unlink(tmpfile)
                return False
                
            playsound(tmpfile)
            os.unlink(tmpfile)
            return True
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return False
    
    def pyttsx_speech(self, text):
        """Use pyttsx3 for offline speech"""
        try:
            if self.pyttsx_engine:
                # Add event callbacks to handle interruptions
                def on_start(name):
                    pass
                    
                def on_word(name, location, length):
                    if self.interrupt_flag:
                        self.pyttsx_engine.stop()
                        
                def on_end(name, completed):
                    pass
                    
                self.pyttsx_engine.connect('started-utterance', on_start)
                self.pyttsx_engine.connect('started-word', on_word)
                self.pyttsx_engine.connect('finished-utterance', on_end)
                
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if sentence.strip():
                        if self.interrupt_flag:
                            break
                        self.pyttsx_engine.say(sentence.strip())
                self.pyttsx_engine.runAndWait()
                return True
            return False
        except Exception as e:
            print(f"pyttsx3 error: {e}")
            return False
    
    def google_tts(self, text):
        """Use Google TTS for decent quality speech"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                
            # Check for interruptions before playing
            if self.interrupt_flag:
                os.unlink(f.name)
                return False
                
            playsound(f.name)
            os.unlink(f.name)
            return True
        except Exception as e:
            print(f"Google TTS error: {e}")
            return False
    
    def text_to_speech(self, text):
        """Convert text to speech using the best available option"""
        if not text.strip():
            return
            
        self.is_speaking = True
        self.interrupt_flag = False
        
        # Start a thread to listen for interruptions while speaking
        interrupt_thread = threading.Thread(target=self.listen_for_interruptions)
        interrupt_thread.daemon = True
        interrupt_thread.start()
        
        # Try Edge TTS first for most natural voice
        try:
            if asyncio.run(self.edge_tts_speech(text)):
                self.is_speaking = False
                return
        except Exception as e:
            print(f"Edge TTS runtime error: {e}")
        
        # If interrupted, don't try other methods
        if self.interrupt_flag:
            self.is_speaking = False
            return
            
        # Try pyttsx3
        if self.pyttsx_speech(text):
            self.is_speaking = False
            return
            
        # If interrupted, don't try other methods
        if self.interrupt_flag:
            self.is_speaking = False
            return
        
        # Fallback to Google TTS
        if self.google_tts(text):
            self.is_speaking = False
            return
        
        # Ultimate fallback
        print(f"AI: {text}")
        try:
            beep = Sine(440).to_audio_segment(duration=100)
            play(beep)
        except:
            pass
        
        self.is_speaking = False
    
    def listen_for_interruptions(self):
        """Listen for user interruptions while the AI is speaking"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 500  # Higher threshold to avoid false positives
        recognizer.dynamic_energy_threshold = True
        
        while self.is_speaking and not self.interrupt_flag:
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    try:
                        # Listen for very short periods to quickly detect interruptions
                        audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=0.5)
                        # If we get audio, set the interrupt flag
                        self.interrupt_flag = True
                        print("💬 Interruption detected!")
                        break
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        continue
                    except Exception as e:
                        print(f"Interruption detection error: {e}")
                        continue
            except Exception as e:
                print(f"Microphone error in interruption detection: {e}")
                continue
    
    def listen_for_speech(self):
        """Listen for speech using microphone"""
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.WaitTimeoutError:
                    return None
                except sr.UnknownValueError:
                    print("Could not understand audio")
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
        
        # Start with a welcome message
        self.text_to_speech("Hello there! I'm ready to have a conversation with you. What would you like to talk about?")
        
        while self.is_listening:
            # Wait if currently speaking
            if self.is_speaking:
                time.sleep(0.1)
                continue
                
            # Reset interrupt flag
            self.interrupt_flag = False
                
            # Listen for speech
            text = self.listen_for_speech()
            
            if text and text.strip():
                print(f"👤 You: {text}")
                
                # Check for exit command
                if text.lower() in ['exit', 'quit', 'stop', 'goodbye', 'end']:
                    self.text_to_speech("It was nice talking with you. Goodbye!")
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

if __name__ == "__main__":
    agent = RealTimeVoiceAgent()
    agent.start_conversation()