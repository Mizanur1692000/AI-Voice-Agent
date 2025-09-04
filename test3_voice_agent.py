import os
import time
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import re
import warnings
import threading
import pyaudio
import requests
import json
from datetime import datetime, timedelta
import urllib.parse
import queue
import io

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
        self.recognizer.pause_threshold = 1.2
        
        self.is_listening = False
        self.is_speaking = False
        self.conversation_history = []
        self.setup_tts()
        
        # For interruption functionality
        self.interrupt_flag = False
        self.audio_thread = None
        self.stop_audio = False
        
        # Audio queue for sequential playback
        self.audio_queue = queue.Queue()
        self.audio_playing = False
        
        # PyAudio for interruptible audio playback
        self.p = pyaudio.PyAudio()
        
        # Start audio playback thread
        self.audio_worker_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_worker_thread.start()
        
    def setup_apis(self):
        """Initialize API clients with keys from environment variables"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure the Gemini API
        genai.configure(api_key=google_api_key)
        
        # Initialize the model with streaming enabled
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,  # Reduced for faster responses
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",  
            generation_config=generation_config,
            safety_settings=safety_settings
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
    
    def get_local_info(self, query):
        """Get local information without API calls"""
        query_lower = query.lower()
        
        # Date and time information
        if any(word in query_lower for word in ["date", "day", "today", "time", "now", "current time"]):
            now = datetime.now()
            
            if "date" in query_lower or "day" in query_lower or "today" in query_lower:
                return f"Today is {now.strftime('%A, %B %d, %Y')}"
            elif "time" in query_lower:
                return f"The current time is {now.strftime('%I:%M %p')}"
            else:
                return f"Today is {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}"
        
        # Simple math calculations
        elif any(word in query_lower for word in ["calculate", "what is", "plus", "minus", "times", "multiplied", "divided"]):
            try:
                # Simple math expression extraction
                if "what is" in query_lower:
                    expression = query_lower.split("what is")[1].strip()
                    expression = re.sub(r'[^\d+\-*/().]', '', expression)
                    result = eval(expression)
                    return f"{expression} equals {result}"
            except:
                pass  # Fall through to API-based response
        
        return None
    
    def get_real_time_info(self, query):
        """Get real-time information using free APIs"""
        # First try local information
        local_info = self.get_local_info(query)
        if local_info:
            return local_info
            
        try:
            # Check for specific types of queries
            query_lower = query.lower()
            
            # Weather queries
            if any(word in query_lower for word in ["weather", "temperature", "forecast", "rain", "sunny"]):
                return self.get_weather_info(query)
            
            # News queries
            elif any(word in query_lower for word in ["news", "headline", "update", "latest"]):
                return self.get_news_info(query)
            
            # Stock queries
            elif any(word in query_lower for word in ["stock", "share", "market", "price"]):
                return self.get_stock_info(query)
            
            # Cricket/sports queries
            elif any(word in query_lower for word in ["cricket", "match", "score", "sports"]):
                return self.get_sports_info(query)
            
            # General queries - use DuckDuckGo
            else:
                return self.search_duckduckgo(query)
                
        except Exception as e:
            print(f"Error getting real-time info: {e}")
            return "I couldn't retrieve real-time information at the moment."
    
    def get_weather_info(self, query):
        """Get weather information using Open-Meteo API (free, no API key)"""
        try:
            # Extract location from query (default to Dhaka)
            location = "Dhaka"
            if "weather in" in query.lower():
                location = query.lower().split("weather in")[1].strip()
            elif "weather at" in query.lower():
                location = query.lower().split("weather at")[1].strip()
            
            # First, get coordinates for the location
            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1"
            geo_response = requests.get(geocoding_url, timeout=5)
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                return f"Could not find weather information for {location}."
            
            latitude = geo_data["results"][0]["latitude"]
            longitude = geo_data["results"][0]["longitude"]
            
            # Get weather data
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,weather_code,wind_speed_10m&timezone=auto"
            weather_response = requests.get(weather_url, timeout=5)
            weather_data = weather_response.json()
            
            if "current" in weather_data:
                temp = weather_data["current"]["temperature_2m"]
                weather_code = weather_data["current"]["weather_code"]
                
                # Convert weather code to description
                weather_descriptions = {
                    0: "clear sky",
                    1: "mainly clear", 
                    2: "partly cloudy",
                    3: "overcast",
                    45: "foggy",
                    48: "depositing rime fog",
                    51: "light drizzle",
                    53: "moderate drizzle",
                    55: "dense drizzle",
                    61: "slight rain",
                    63: "moderate rain",
                    65: "heavy rain",
                    80: "slight rain showers",
                    81: "moderate rain showers",
                    82: "violent rain showers",
                    95: "thunderstorm",
                    96: "thunderstorm with slight hail",
                    99: "thunderstorm with heavy hail"
                }
                
                condition = weather_descriptions.get(weather_code, "unknown conditions")
                return f"Current weather in {location}: {temp}°C with {condition}."
            else:
                return f"Weather information not available for {location}."
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self.search_duckduckgo(f"weather in {location}")
    
    def get_news_info(self, query):
        """Get news information using DuckDuckGo (no API key needed)"""
        try:
            # Extract topic from query
            topic = "news"
            if "sports" in query.lower():
                topic = "sports"
            elif "technology" in query.lower() or "tech" in query.lower():
                topic = "technology"
            elif "business" in query.lower():
                topic = "business"
            elif "entertainment" in query.lower():
                topic = "entertainment"
            elif "health" in query.lower():
                topic = "health"
            
            # Use DuckDuckGo for news
            return self.search_duckduckgo(f"latest {topic} news")
                
        except Exception as e:
            print(f"News API error: {e}")
            return self.search_duckduckgo("latest news")
    
    def get_stock_info(self, query):
        """Get stock information using Alpha Vantage API (free tier)"""
        try:
            # Use a demo API key (might have limited requests)
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
            
            # Extract stock symbol from query
            symbol = "AAPL"  # Default to Apple
            if "stock" in query.lower():
                words = query.lower().split()
                for i, word in enumerate(words):
                    if word == "stock" and i < len(words) - 1:
                        symbol = words[i + 1].upper()
                        break
            
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                price = data["Global Quote"]["05. price"]
                change = data["Global Quote"]["09. change"]
                change_percent = data["Global Quote"]["10. change percent"]
                return f"Current stock price for {symbol}: ${price}, change: {change} ({change_percent})."
            else:
                return f"Stock information not available for {symbol}. Please try a different stock symbol."
                
        except Exception as e:
            print(f"Stock API error: {e}")
            return self.search_duckduckgo(f"{symbol} stock price")
    
    def get_sports_info(self, query):
        """Get sports information using free APIs"""
        try:
            # For cricket, use CricAPI (free tier available)
            if "cricket" in query.lower():
                api_key = os.getenv("CRICAPI_KEY", "your_cricapi_key_here")
                
                if api_key == "your_cricapi_key_here":
                    # If no API key, use DuckDuckGo search
                    return self.search_duckduckgo("latest cricket news")
                
                url = f"https://cricapi.com/api/matches?apikey={api_key}"
                response = requests.get(url, timeout=5)
                data = response.json()
                
                if data.get("matches"):
                    # Find the most recent match
                    for match in data["matches"]:
                        if match.get("matchStarted", False) and not match.get("matchEnded", True):
                            team1 = match.get("team-1", "")
                            team2 = match.get("team-2", "")
                            return f"Current cricket match: {team1} vs {team2} is in progress."
                    
                    return "No live cricket matches at the moment."
                else:
                    return "Cricket information not available at the moment."
            
            # For other sports, use DuckDuckGo
            else:
                return self.search_duckduckgo(query)
                
        except Exception as e:
            print(f"Sports API error: {e}")
            return self.search_duckduckgo(query)
    
    def search_duckduckgo(self, query):
        """Search using DuckDuckGo Instant Answer API (free, no API key)"""
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get("AbstractText"):
                return data["AbstractText"]
            elif data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                return data["RelatedTopics"][0].get("Text", "No information found")
            else:
                return "No information found for your query."
        except Exception as e:
            print(f"DuckDuckGo error: {e}")
            return "I couldn't retrieve information at the moment."
    
    def generate_response(self, user_input):
        """Generate a response using Gemini"""
        # Get real-time information
        real_time_response = self.get_real_time_info(user_input)
        
        # Build the prompt with conversation history
        system_prompt = """You're a helpful, friendly AI assistant, always ready for a casual, natural conversation. 
        Keep responses clear, concise, and conversational, with a tone that feels like chatting with a real person.
        Do not use any emojis, markdown formatting, asterisks, bullets, or other special characters in your response.
        If you have real-time information, incorporate it naturally into your response."""
        
        # Format conversation history
        history_text = ""
        for msg in self.conversation_history[-4:]:  # Reduced history for faster processing
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['content']}\n"
        
        # Add real-time information
        enhanced_input = f"{user_input}\n\nReal-time information: {real_time_response}"
        
        full_prompt = f"{system_prompt}\n\n{history_text}User: {enhanced_input}\nAssistant:"
        
        try:
            # Generate content (non-streaming for faster response)
            response = self.model.generate_content(full_prompt)
            
            # Clean up response for better TTS
            cleaned_response = self.clean_text_for_speech(response.text)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
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
    
    def google_tts_to_audio(self, text):
        """Convert text to audio using Google TTS"""
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file in memory
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Load audio from buffer
            audio_segment = AudioSegment.from_mp3(audio_buffer)
            
            return audio_segment
        except Exception as e:
            print(f"Google TTS error: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech and add to queue"""
        if not text.strip():
            return
            
        # Add to audio queue
        self.audio_queue.put(text)
    
    def audio_worker(self):
        """Worker thread that processes audio queue sequentially"""
        while True:
            try:
                text = self.audio_queue.get()
                if text is None:  # Sentinel value to stop the thread
                    break
                    
                self.is_speaking = True
                self.stop_audio = False
                self.interrupt_flag = False
                
                # Start interruption detection in a separate thread
                interrupt_thread = threading.Thread(target=self.listen_for_interruptions)
                interrupt_thread.daemon = True
                interrupt_thread.start()
                
                # Use Google TTS
                audio_segment = self.google_tts_to_audio(text)
                
                # If interrupted, don't try other methods
                if self.interrupt_flag:
                    self.is_speaking = False
                    continue
                    
                # If we have audio, play it with interruption support
                if audio_segment is not None:
                    success = self.play_audio_interruptible(audio_segment)
                    if not success:
                        print("Audio playback was interrupted")
                
                self.is_speaking = False
                self.audio_queue.task_done()
                
            except Exception as e:
                print(f"Audio worker error: {e}")
                self.is_speaking = False
    
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
        print("🌐 Real-time information available for sports, news, weather, and stocks")
        
        # Start with a welcome message
        welcome_text = "Hello there!"
        print(f"🤖 AI: {welcome_text}")
        self.text_to_speech(welcome_text)
        
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
                    goodbye_text = "Goodbye!"
                    print(f"🤖 AI: {goodbye_text}")
                    self.text_to_speech(goodbye_text)
                    # Wait for goodbye message to finish
                    time.sleep(2)
                    break
                
                # Generate response
                response = self.generate_response(text)
                print(f"🤖 AI: {response}")
                
                # Add response to audio queue
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