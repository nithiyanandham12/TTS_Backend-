from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import base64
import logging
import io
import wave
from sarvamai import SarvamAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware - must be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
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


def split_text_into_chunks(text, max_length=1500):
    """
    Split text into chunks at sentence boundaries, respecting max_length
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(sentence) + 2 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
        else:
            if current_chunk:
                current_chunk += " " + sentence + "."
            else:
                current_chunk = sentence + "."
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If any chunk is still too long, split it by character limit
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Split by character limit at word boundaries
            words = chunk.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 > max_length:
                    if temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                    temp_chunk = word
                else:
                    temp_chunk += " " + word if temp_chunk else word
            if temp_chunk:
                final_chunks.append(temp_chunk.strip())
    
    return final_chunks


def concatenate_wav_files(wav_data_list):
    """
    Concatenate multiple WAV file byte arrays into a single WAV file
    """
    if not wav_data_list:
        return None
    
    if len(wav_data_list) == 1:
        return wav_data_list[0]
    
    # Read all WAV files
    wav_objects = []
    sample_rate = None
    sample_width = None
    n_channels = None
    
    for wav_data in wav_data_list:
        wav_file = wave.open(io.BytesIO(wav_data), 'rb')
        wav_objects.append(wav_file)
        
        # Verify all have same parameters
        if sample_rate is None:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            n_channels = wav_file.getnchannels()
        else:
            if (wav_file.getframerate() != sample_rate or 
                wav_file.getsampwidth() != sample_width or 
                wav_file.getnchannels() != n_channels):
                logger.warning("WAV files have different parameters, may cause issues")
    
    # Create output WAV file
    output = io.BytesIO()
    output_wav = wave.open(output, 'wb')
    output_wav.setnchannels(n_channels)
    output_wav.setsampwidth(sample_width)
    output_wav.setframerate(sample_rate)
    
    # Write all frames
    for wav_file in wav_objects:
        frames = wav_file.readframes(wav_file.getnframes())
        output_wav.writeframes(frames)
        wav_file.close()
    
    output_wav.close()
    return output.getvalue()


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using SarvamAI
    Returns audio file as WAV format
    Supports up to 1500 characters per API call - automatically chunks longer text
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        text_length = len(request.text)
        logger.info(f"Total text length: {text_length} characters")
        logger.info(f"Parameters: lang={request.target_language_code}, speaker={request.speaker}")
        
        # SarvamAI API limit is 1500 characters per request
        # But to be safe, we'll use 1200 to account for any API-side truncation
        MAX_CHUNK_LENGTH = 1200
        
        # Split text into chunks if needed
        text_chunks = split_text_into_chunks(request.text, MAX_CHUNK_LENGTH)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        if len(text_chunks) > 1:
            logger.info(f"Text exceeds {MAX_CHUNK_LENGTH} characters, processing {len(text_chunks)} chunks")
            for i, chunk in enumerate(text_chunks, 1):
                logger.info(f"Chunk {i}/{len(text_chunks)}: {len(chunk)} characters - {chunk[:100]}...")
        
        # Process each chunk
        audio_chunks = []
        for i, chunk in enumerate(text_chunks, 1):
            logger.info(f"Processing chunk {i}/{len(text_chunks)} ({len(chunk)} characters)")
            logger.info(f"Chunk {i} full text: {chunk}")
            logger.info(f"Chunk {i} first 200 chars: {chunk[:200]}")
            logger.info(f"Chunk {i} last 200 chars: {chunk[-200:]}")
            
            response = client.text_to_speech.convert(
                text=chunk,
                target_language_code=request.target_language_code,
                speaker=request.speaker,
                pitch=request.pitch,
                pace=request.pace,
                loudness=request.loudness,
                speech_sample_rate=request.speech_sample_rate,
                enable_preprocessing=request.enable_preprocessing,
                model=request.model
            )
            
            logger.info(f"Chunk {i} API Response received")
            
            # Extract audio from response
            chunk_audio_bytes = None
            
            # Log response structure for debugging
            logger.info(f"Chunk {i}: Response type: {type(response)}")
            logger.info(f"Chunk {i}: Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            
            # Try different ways to access audio data
            if hasattr(response, 'audio_content'):
                chunk_audio_bytes = response.audio_content
                logger.info(f"Chunk {i}: Found audio_content attribute, size: {len(chunk_audio_bytes) if chunk_audio_bytes else 0} bytes")
            elif hasattr(response, 'audios'):
                audios = response.audios
                logger.info(f"Chunk {i}: Found audios attribute, count: {len(audios) if audios else 0}")
                if audios and len(audios) > 0:
                    # Check if there are multiple audio segments
                    if len(audios) > 1:
                        logger.info(f"Chunk {i}: Multiple audio segments found ({len(audios)}), concatenating them")
                        # Concatenate multiple audio segments for this chunk
                        segment_audios = []
                        for j, audio_seg in enumerate(audios):
                            if isinstance(audio_seg, str):
                                seg_bytes = base64.b64decode(audio_seg)
                            else:
                                seg_bytes = audio_seg
                            segment_audios.append(seg_bytes)
                            logger.info(f"Chunk {i}, Segment {j+1}: {len(seg_bytes)} bytes")
                        # Concatenate segments for this chunk
                        if len(segment_audios) > 1:
                            chunk_audio_bytes = concatenate_wav_files(segment_audios)
                        else:
                            chunk_audio_bytes = segment_audios[0]
                    else:
                        # Single audio segment
                        if isinstance(audios[0], str):
                            chunk_audio_bytes = base64.b64decode(audios[0])
                        else:
                            chunk_audio_bytes = audios[0]
                logger.info(f"Chunk {i}: Extracted audio from audios attribute")
            elif hasattr(response, 'audio'):
                chunk_audio_bytes = response.audio
                logger.info(f"Chunk {i}: Found audio attribute, size: {len(chunk_audio_bytes) if chunk_audio_bytes else 0} bytes")
            elif hasattr(response, 'data'):
                data = response.data
                logger.info(f"Chunk {i}: Found data attribute, type: {type(data)}")
                if isinstance(data, dict) and 'audios' in data:
                    audios_list = data['audios']
                    logger.info(f"Chunk {i}: Found audios in data, count: {len(audios_list) if audios_list else 0}")
                    if audios_list and len(audios_list) > 0:
                        if len(audios_list) > 1:
                            # Multiple segments
                            segment_audios = []
                            for j, audio_seg in enumerate(audios_list):
                                if isinstance(audio_seg, str):
                                    seg_bytes = base64.b64decode(audio_seg)
                                else:
                                    seg_bytes = audio_seg
                                segment_audios.append(seg_bytes)
                            chunk_audio_bytes = concatenate_wav_files(segment_audios)
                        else:
                            audio_base64 = audios_list[0]
                            chunk_audio_bytes = base64.b64decode(audio_base64) if isinstance(audio_base64, str) else audio_base64
                logger.info(f"Chunk {i}: Extracted audio from data attribute")
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
                    
                    logger.info(f"Chunk {i}: Response dict keys: {list(response_dict.keys())}")
                    
                    if 'audios' in response_dict:
                        audios_list = response_dict['audios']
                        logger.info(f"Chunk {i}: Found audios in dict, count: {len(audios_list) if audios_list else 0}")
                        if audios_list and len(audios_list) > 0:
                            if len(audios_list) > 1:
                                # Multiple segments
                                segment_audios = []
                                for j, audio_seg in enumerate(audios_list):
                                    if isinstance(audio_seg, str):
                                        seg_bytes = base64.b64decode(audio_seg)
                                    else:
                                        seg_bytes = audio_seg
                                    segment_audios.append(seg_bytes)
                                chunk_audio_bytes = concatenate_wav_files(segment_audios)
                            else:
                                audio_base64 = audios_list[0]
                                chunk_audio_bytes = base64.b64decode(audio_base64) if isinstance(audio_base64, str) else audio_base64
                            logger.info(f"Chunk {i}: Converted response to dict and found audios")
                        elif 'audio_content' in response_dict:
                            chunk_audio_bytes = response_dict['audio_content']
                            logger.info(f"Chunk {i}: Found audio_content in dict")
                except Exception as e:
                    logger.error(f"Chunk {i}: Could not convert response to dict: {e}")
                    logger.exception(e)
            
            if chunk_audio_bytes:
                audio_chunks.append(chunk_audio_bytes)
                logger.info(f"Chunk {i}: Successfully extracted {len(chunk_audio_bytes)} bytes of audio")
            else:
                logger.error(f"Chunk {i}: Could not extract audio from response")
                logger.error(f"Chunk {i}: Response object: {response}")
                raise HTTPException(status_code=500, detail=f"Could not extract audio data from chunk {i}")
        
        # Concatenate all audio chunks
        if len(audio_chunks) > 1:
            logger.info(f"Concatenating {len(audio_chunks)} audio chunks")
            audio_bytes = concatenate_wav_files(audio_chunks)
            logger.info(f"Concatenated audio size: {len(audio_bytes)} bytes")
        else:
            audio_bytes = audio_chunks[0] if audio_chunks else None
        
        if audio_bytes:
            audio_size = len(audio_bytes)
            logger.info(f"Successfully processed audio, total size: {audio_size} bytes")
            logger.info(f"Text processed length: {text_length} characters")
            logger.info(f"Number of chunks processed: {len(text_chunks)}")
            logger.info(f"Audio size: {audio_size} bytes ({audio_size / 1024:.2f} KB)")
            
            # Estimate audio duration (rough calculation: WAV file size / (sample_rate * channels * bytes_per_sample))
            # For 22050 Hz, mono, 16-bit: duration ≈ size / (22050 * 1 * 2)
            estimated_duration = audio_size / (request.speech_sample_rate * 1 * 2) if request.speech_sample_rate else 0
            logger.info(f"Estimated audio duration: {estimated_duration:.2f} seconds")
            
            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "X-Text-Length": str(text_length),
                    "X-Audio-Size": str(audio_size),
                    "X-Estimated-Duration": str(round(estimated_duration, 2)),
                    "X-Chunks-Processed": str(len(text_chunks))
                }
            )
        else:
            logger.error(f"Could not extract audio from response")
            raise HTTPException(status_code=500, detail="Could not extract audio data from TTS response")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS conversion failed with exception:")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "TTS API is running"}

