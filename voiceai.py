from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    groq,
    azure
)

import logging

import asyncio
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""You are a helpful multilingual voice AI assistant. 
            You understand English and Hindi/Urdu speech input.
            
            CRITICAL LANGUAGE HANDLING RULES:
            1. If the user speaks in English, respond ONLY in English
            2. If the user speaks in Urdu (which appears as Hindi text), respond ONLY in Urdu script (Arabic script)
            3. Choose ONE language per response - never mix languages
            4. ALWAYS acknowledge what the user said and provide a helpful response
        
            
            LANGUAGE DETECTION:
            - English input = English text in Latin script
            - Urdu input = Hindi text in Devanagari script (आप, कैसे, etc.) - this represents Urdu speech transcribed as Hindi
            
            URDU RESPONSE GUIDELINES (when input is Hindi script):
            - Write in proper Urdu script using Arabic alphabet
            - Use Pakistani Urdu vocabulary and expressions naturally
            - Use respectful forms like "آپ" (aap) 
            - Use Pakistani Urdu expressions like "بالکل" (bilkul), "ضرور" (zaroor), "ماشاءاللہ" (mashallah)
            - Use "میں" (main) for "I"
            - Use respectful forms and Pakistani cultural references
            - Keep responses natural and conversational
            - Always use short, clear sentences for better TTS pronunciation
            
            Examples:
            - User (English): "How are you?" → Response (English): "I'm doing great! How can I help you today?"
            - User (Hindi): "आप कैसे हैं?" → Response (Urdu): "میں بالکل ٹھیک ہوں! آپ کیسے ہیں؟ کیا مدد چاہیے؟"
            - User (Hindi): "क्या हाल है?" → Response (Urdu): "سب ٹھیک ہے! آپ بتائیے کیا کام ہے؟"
            - User (English): "Hello" → Response (English): "Hello! How can I assist you today?"
            - User (Hindi): "नमस्ते" → Response (Urdu): "آداب! میں آپ کی کیا مدد کر سکتا ہوں؟"
            
            IMPORTANT: 
            - Hindi script input (Devanagari) = User spoke Urdu → Respond in Urdu script
            - English script input = User spoke English → Respond in English
            - Be natural, helpful, and culturally appropriate for Pakistani Urdu speakers
            - Always speak in short sentences for optimal TTS performance""")
        
        self.user_language_preference = "english"  # default
        self.conversation_history = []
    
    def _is_hindi_input(self, user_input: str) -> bool:
        """Detect if the input is in Hindi script (representing Urdu speech)"""
        try:
            if not user_input or user_input.strip() == "":
                return False
            
            # Check for Hindi/Devanagari script - this indicates Urdu speech transcribed as Hindi
            has_devanagari = any('\u0900' <= char <= '\u097F' for char in user_input)
            return has_devanagari
                
        except Exception as e:
            logger.warning(f"Hindi detection error: {e}")
            return False
    
    def _is_urdu_script(self, text: str) -> bool:
        """Check if text is already in Urdu/Arabic script"""
        try:
            if not text or text.strip() == "":
                return False
            
            # Check for Arabic/Urdu script
            has_arabic_script = any(
                ('\u0600' <= char <= '\u06FF') or  # Arabic
                ('\u0750' <= char <= '\u077F') or  # Arabic Supplement  
                ('\u08A0' <= char <= '\u08FF') or  # Arabic Extended-A
                ('\uFB50' <= char <= '\uFDFF') or  # Arabic Presentation Forms-A
                ('\uFE70' <= char <= '\uFEFF')     # Arabic Presentation Forms-B
                for char in text
            )
            
            return has_arabic_script
        except Exception as e:
            logger.warning(f"Urdu script detection error: {e}")
            return False
    
    def _should_respond_in_urdu(self, user_input: str) -> bool:
        """Determine if response should be in Urdu script"""
        # Respond in Urdu if user input is Hindi (representing Urdu speech)
        return self._is_hindi_input(user_input)
    
    async def say(self, message: str, *, allow_interruptions: bool = True, add_to_chat_ctx: bool = True):
        """Override say method to ensure consistent language responses"""
        try:
            # Detect response language for logging
            is_hindi_input = self._is_hindi_input(message)  # Check if this was from Hindi input
            is_urdu_response = self._is_urdu_script(message)  # Check if response is in Urdu script
            
            if is_urdu_response:
                lang_indicator = "Urdu"
            elif is_hindi_input:
                lang_indicator = "English (from Hindi input)"
            else:
                lang_indicator = "English"
                
            logger.info(f"Assistant responding in {lang_indicator}: {message[:100]}...")
            
            return await super().say(message, allow_interruptions=allow_interruptions, add_to_chat_ctx=add_to_chat_ctx)
        except Exception as e:
            logger.error(f"Error in say method: {e}")
            try:
                fallback_msg = "I'm having trouble responding. Please try again."
                return await super().say(fallback_msg, 
                                       allow_interruptions=allow_interruptions, 
                                       add_to_chat_ctx=add_to_chat_ctx)
            except:
                pass


    
    


async def entrypoint(ctx: agents.JobContext):
    try:
        logger.info("Starting agent initialization...")
        
        # Validate environment variables
        azure_key = os.getenv('AZURE_SPEECH_KEY')
        azure_region = os.getenv('AZURE_SPEECH_REGION')
        groq_key = os.getenv('GROQ_API_KEY')
        deepgram_key = os.getenv('DEEPGRAM_API_KEY')
        
        if not azure_key or not azure_region:
            logger.error("Azure Speech credentials not found in environment variables")
            return
        
        if not groq_key:
            logger.error("Groq API key not found in environment variables")
            return
            
        if not deepgram_key:
            logger.error("Deepgram API key not found in environment variables")
            return
        
        assistant = Assistant()
        logger.info("Assistant created successfully")
        
        # Initialize components with better error handling
        try:
            stt = groq.STT(
                model="whisper-large-v3-turbo", 
                api_key=groq_key,
                language='auto',
                detect_language=True       
               
            )
            logger.info("STT initialized")
        except Exception as e:
            try:
                stt = deepgram.STT(
                    model="nova-3",  # Changed to nova-2 for better stability
                    language='multi',
                    interim_results=True,  # Enable interim results for better responsiveness
                    smart_format=True,     # Enable smart formatting
                    punctuate=True,        # Enable punctuation
                
                )
                logger.info("STT initialized")
            except Exception as e:
                logger.error(f"STT initialization failed: {e}")
                return
                
            
        
        try:
            llm = groq.LLM(model="qwen/qwen3-32b")  # Optimized for Urdu and regional languages
            logger.info("LLM initialized with Urdu-optimized model")
        except Exception as e:
            logger.warning(f"Urdu-optimized LLM failed, trying fallback: {e}")
            try:
                llm = groq.LLM(model="llama-3.3-70b-versatile")  # Fallback model
                logger.info("LLM initialized with fallback model")
            except Exception as fallback_e:
                logger.error(f"LLM initialization failed completely: {fallback_e}")
                return
        
        try:
            tts = azure.TTS(
                speech_key=azure_key,
                speech_region=azure_region,
                voice="ur-PK-AsadNeural",  # Hindi voice for better Hindi pronunciation
                language="ur-PK",  # Set Hindi as default language
                # Add fallback options
                sample_rate=24000,
            )
            logger.info("TTS initialized with Hindi voice")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            # Fallback to English voice
            try:
                tts = azure.TTS(
                    speech_key=azure_key,
                    speech_region=azure_region,
                    voice="en-US-AriaNeural",
                    sample_rate=24000
                )
                logger.info("TTS initialized with fallback English voice")
            except Exception as fallback_e:
                logger.error(f"TTS fallback failed: {fallback_e}")
                return
        
        try:
            vad = silero.VAD.load(
                # Configure VAD for better multilingual detection
                min_silence_duration=0.5,  # Reduced for more responsive detection
                min_speech_duration=0.3,   # Reduced for shorter utterances
            )
            logger.info("VAD initialized with multilingual settings")
        except Exception as e:
            # Fallback to default VAD if custom settings fail
            try:
                vad = silero.VAD.load()
                logger.info("VAD initialized with default settings")
            except Exception as fallback_e:
                logger.error(f"VAD initialization failed completely: {fallback_e}")
                return
        
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            
        )
        
        logger.info("Session created, starting...")
        
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(), 
            ),
        )
        
        logger.info("Session started successfully - Hindi input → Urdu script output pipeline")
        
        # Generate initial greeting with timeout
        try:
            await asyncio.wait_for(
                session.generate_reply(
                    instructions="Greet the user warmly in English, mentioning that you can speak both English and Urdu, and ask how you can help them today."
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Initial greeting generation timed out")
        
    except Exception as e:
        logger.error(f"Entrypoint error: {e}")
        raise



if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

