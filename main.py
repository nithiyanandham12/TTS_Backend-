from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import base64
import logging
from sarvamai import SarvamAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SarvamAI client
api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise ValueError("SARVAM_API_KEY not found in environment variables")

client = SarvamAI(api_subscription_key=api_key)


class TTSRequest(BaseModel):
    text: str
    target_language_code: str = "hi-IN"
    speaker: str = "anushka"
    pitch: float = 0
    pace: float = 1
    loudness: float = 1
    speech_sample_rate: int = 22050
    enable_preprocessing: bool = True
    model: str = "bulbul:v2"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using SarvamAI
    Returns audio file as WAV format
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Converting text to speech: {request.text[:50]}...")
        logger.info(f"Parameters: lang={request.target_language_code}, speaker={request.speaker}")
        
        response = client.text_to_speech.convert(
            text=request.text,
            target_language_code=request.target_language_code,
            speaker=request.speaker,
            pitch=request.pitch,
            pace=request.pace,
            loudness=request.loudness,
            speech_sample_rate=request.speech_sample_rate,
            enable_preprocessing=request.enable_preprocessing,
            model=request.model
        )
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {dir(response)}")
        
        # Handle TextToSpeechResponse object
        audio_bytes = None
        
        # Try different ways to access audio data
        if hasattr(response, 'audio_content'):
            audio_bytes = response.audio_content
            logger.info("Found audio_content attribute")
        elif hasattr(response, 'audios'):
            audios = response.audios
            if audios and len(audios) > 0:
                # If it's base64 encoded string
                if isinstance(audios[0], str):
                    audio_bytes = base64.b64decode(audios[0])
                else:
                    audio_bytes = audios[0]
            logger.info("Found audios attribute")
        elif hasattr(response, 'audio'):
            audio_bytes = response.audio
            logger.info("Found audio attribute")
        elif hasattr(response, 'data'):
            data = response.data
            if isinstance(data, dict) and 'audios' in data:
                audio_base64 = data['audios'][0] if len(data['audios']) > 0 else None
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
            logger.info("Found data attribute")
        else:
            # Try to convert to dict if possible (Pydantic v2 uses model_dump)
            try:
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                elif hasattr(response, 'dict'):
                    response_dict = response.dict()
                elif hasattr(response, '__dict__'):
                    response_dict = response.__dict__
                else:
                    response_dict = dict(response)
                
                logger.info(f"Response dict keys: {list(response_dict.keys())}")
                
                if 'audios' in response_dict and len(response_dict['audios']) > 0:
                    audio_base64 = response_dict['audios'][0]
                    audio_bytes = base64.b64decode(audio_base64)
                    logger.info("Converted response to dict and found audios")
                elif 'audio_content' in response_dict:
                    audio_bytes = response_dict['audio_content']
                    logger.info("Found audio_content in dict")
            except Exception as e:
                logger.error(f"Could not convert response to dict: {e}")
        
        if audio_bytes:
            logger.info(f"Successfully extracted audio, size: {len(audio_bytes)} bytes")
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        else:
            logger.error(f"Could not extract audio from response. Response: {response}")
            raise HTTPException(status_code=500, detail="Could not extract audio data from TTS response")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS conversion failed with exception:")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "TTS API is running"}

