"""Interactive Avatar WebSocket Server."""
import asyncio
import logging
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch.distributed as dist

from services import STTService, LLMService, TTSService, AvatarService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Interactive LiveAvatar Server")

# Global services
stt_service: Optional[STTService] = None
llm_service: Optional[LLMService] = None
tts_service: Optional[TTSService] = None
avatar_service: Optional[AvatarService] = None

# Session management
active_sessions: Dict[str, dict] = {}

# Distributed setup
global_rank: int = 0
world_size: int = 1


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global stt_service, llm_service, tts_service, avatar_service, global_rank, world_size
    
    if dist.is_initialized():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Distributed mode detected: Rank {global_rank}/{world_size}")
    else:
        logger.info("Single-process mode.")

    if global_rank == 0:
        logger.info("Starting Interactive Avatar Server (Rank 0)...")
        logger.info(f"OpenAI Model: {settings.openai_model}")
        logger.info(f"LiveAvatar Config: {settings.liveavatar_size} @ {settings.liveavatar_sample_steps} steps")
        
        # Initialize services
        logger.info("Initializing STT service...")
        stt_service = STTService()
        
        logger.info("Initializing LLM service...")
        llm_service = LLMService()
        
        logger.info("Initializing TTS service...")
        tts_service = TTSService()
        
        logger.info("Initializing Avatar service (this may take a while)...")
        avatar_service = AvatarService()
        await avatar_service.initialize()
        
        logger.info("All services initialized successfully on Rank 0!")
        
        # Mount static files for web interface
        if os.path.exists("web_interface"):
            app.mount("/static", StaticFiles(directory="web_interface"), name="static")
    else:
        logger.info(f"Rank {global_rank}: Not initializing API services, waiting for worker loop to start.")


@app.get("/")
async def get_index():
    """Serve the main web interface."""
    index_path = Path("web_interface/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content="<h1>Interactive Avatar Server Running</h1><p>Web interface not found. Please create web_interface/index.html</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "stt": stt_service is not None,
            "llm": llm_service is not None,
            "tts": tts_service is not None,
            "avatar": avatar_service is not None and avatar_service.is_initialized
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for interactive avatar."""
    await websocket.accept()
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info(f"New WebSocket connection: {session_id}")
    
    # Initialize session
    active_sessions[session_id] = {
        "conversation_history": [],
        "websocket": websocket
    }
    
    try:
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "message": "Connected to Interactive Avatar"
        })
        
        while True:
            # Receive message from client
            message = await websocket.receive()
            
            if "text" in message:
                # Handle JSON messages
                data = json.loads(message["text"])
                await handle_message(session_id, data, websocket)
            
            elif "bytes" in message:
                # Handle binary audio data
                audio_data = message["bytes"]
                await handle_audio(session_id, audio_data, websocket)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")


async def handle_message(session_id: str, data: dict, websocket: WebSocket):
    """Handle text messages from client."""
    message_type = data.get("type")
    
    if message_type == "text_input":
        # Direct text input (bypass STT)
        user_text = data.get("text", "")
        await process_user_input(session_id, user_text, websocket)
    
    elif message_type == "config":
        # Update configuration
        logger.info(f"Config update for {session_id}: {data}")
        await websocket.send_json({
            "type": "config_updated",
            "message": "Configuration updated"
        })


async def handle_audio(session_id: str, audio_data: bytes, websocket: WebSocket):
    """Handle audio data from client."""
    try:
        logger.info(f"Received audio from {session_id}: {len(audio_data)} bytes")
        
        # Send status update
        await websocket.send_json({
            "type": "status",
            "status": "transcribing",
            "message": "Transcribing your speech..."
        })
        
        # Transcribe audio
        user_text = await stt_service.transcribe(audio_data)
        logger.info(f"Transcription: {user_text}")
        
        # Send transcription to client
        await websocket.send_json({
            "type": "transcription",
            "text": user_text
        })
        
        # Process the transcribed text
        await process_user_input(session_id, user_text, websocket)
        
    except Exception as e:
        logger.error(f"Audio handling error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Audio processing failed: {str(e)}"
        })


async def process_user_input(session_id: str, user_text: str, websocket: WebSocket):
    """Process user input and generate avatar response."""
    try:
        session = active_sessions[session_id]
        conversation_history = session["conversation_history"]
        
        # Update status
        await websocket.send_json({
            "type": "status",
            "status": "thinking",
            "message": "Generating response..."
        })
        
        # Generate LLM response
        llm_response = await llm_service.generate_response(
            user_message=user_text,
            conversation_history=conversation_history
        )
        logger.info(f"LLM response: {llm_response}")
        
        # Send text response to client
        await websocket.send_json({
            "type": "response",
            "text": llm_response
        })
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": llm_response})
        
        # Keep only last 10 messages
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        session["conversation_history"] = conversation_history
        
        # Update status
        await websocket.send_json({
            "type": "status",
            "status": "synthesizing",
            "message": "Synthesizing speech..."
        })
        
        # Generate TTS audio
        audio_path = await tts_service.synthesize(llm_response)
        logger.info(f"TTS audio generated: {audio_path}")
        
        # Calculate required video clips based on audio duration
        import wave
        import contextlib
        try:
            # Pydub/librosa might not be installed, use soundfile if available or fallback
            import soundfile as sf
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
        except:
            # Fallback estimation based on character count (1 char ~ 0.08s speaking time)
            duration = len(llm_response) * 0.08
            
        # Each clip is frames/fps seconds long (e.g. 32/48 = 0.67s)
        clip_duration = settings.liveavatar_infer_frames / 48.0
        num_clips = max(1, int(duration / clip_duration) + 1)
        logger.info(f"Audio duration: {duration:.2f}s, requested clips: {num_clips} (at {clip_duration:.2f}s each)")
        
        # Update status
        await websocket.send_json({
            "type": "status",
            "status": "generating_video",
            "message": f"Generating {num_clips} video clips..."
        })
        
        # Generate avatar video
        video_path = await avatar_service.generate_avatar_video(
            audio_path=audio_path,
            num_clips=num_clips
        )
        logger.info(f"Avatar video generated: {video_path}")
        
        # Send video path to client
        await websocket.send_json({
            "type": "video_ready",
            "video_url": f"/video/{Path(video_path).name}",
            "message": "Avatar video ready!"
        })
        
        # Clean up temp audio file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": f"Processing failed: {str(e)}"
        })


@app.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve generated video files."""
    video_path = Path("output/interactive") / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


def worker_loop():
    """
    Worker loop for non-rank-0 processes in distributed mode.
    They continuously wait for broadcast signals from rank 0 and participate in computation.
    """
    global_rank = dist.get_rank()
    logger.info(f"Rank {global_rank} initializing AvatarService...")
    
    # Initialize avatar service for the worker
    service = AvatarService()
    # Synchronously initialize since we are not in an asyncio event loop here
    service._initialize_sync()
    
    logger.info(f"Rank {global_rank} entering worker loop, waiting for inference requests...")
    
    while True:
        try:
            # Wait for broadcast from rank 0
            # Format: [command, audio_path, prompt, reference_image, num_clips, seed]
            inputs = [None] * 6
            dist.broadcast_object_list(inputs, src=0)
            
            command = inputs[0]
            
            if command == "infer":
                _, audio_path, prompt, reference_image, num_clips, seed = inputs
                logger.info(f"[Rank {global_rank}] Received inference request for {num_clips} clips")
                
                # Participate in computation synchronously
                service._generate_sync(
                    audio_path=audio_path,
                    prompt=prompt,
                    reference_image=reference_image,
                    num_clips=num_clips,
                    seed=seed
                )
                logger.info(f"[Rank {global_rank}] Generation completed")
                
            elif command == "idle":
                logger.debug(f"[Rank {global_rank}] Received idle signal, continuing to wait...")
                
        except Exception as e:
            logger.error(f"[Rank {global_rank}] Error in worker loop: {e}")
            import traceback
            traceback.print_exc()
            # Don't break, keep trying to listen
            import time
            time.sleep(1)


if __name__ == "__main__":
    # Initialize PyTorch distributed if torchrun is used
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Initialized distributed group: rank {rank}/{world_size}")
        
        if rank == 0:
            # Rank 0 runs the web server
            logger.info("Rank 0: Starting API server...")
            uvicorn.run(app, host=settings.server_host, port=settings.server_port)
        else:
            # Ranks > 0 run the infinite worker loop
            worker_loop()
            
    else:
        # Single process mode
        logger.info("Starting API server in single-process mode...")
        uvicorn.run(app, host=settings.server_host, port=settings.server_port)
