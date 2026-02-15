# RunPod Setup Guide - Interactive Avatar

## Step 1: Connect to RunPod

Open a terminal on your Mac and run:

```bash
ssh root@195.26.233.54 -p 39340 -i ~/.ssh/id_ed25519
```

## Step 2: Navigate to LiveAvatar

```bash
cd /workspace/LiveAvatar
pwd  # Should show: /workspace/LiveAvatar
```

## Step 3: Create .env File

Copy and paste this entire block:

```bash
cat > .env << 'EOF'
# API Keys
OPENAI_API_KEY=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
ELEVENLABS_API_KEY=sk_4aadaf6c3e29330388e9f808b6a9f1b3740d3617a71008bb

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# LiveAvatar Configuration (Optimized for 15-22 FPS)
LIVEAVATAR_CKPT_DIR=ckpt/Wan2.2-S2V-14B/
LIVEAVATAR_LORA_PATH=Quark-Vision/Live-Avatar
LIVEAVATAR_SIZE=640*360
LIVEAVATAR_INFER_FRAMES=32
LIVEAVATAR_SAMPLE_STEPS=2
ENABLE_FP8=true
ENABLE_COMPILE=true

# OpenAI Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_WHISPER_MODEL=whisper-1
OPENAI_MAX_TOKENS=150

# ElevenLabs Configuration
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_MODEL_ID=eleven_monolingual_v1

# Avatar Configuration
DEFAULT_AVATAR_IMAGE=examples/anchor.jpg
DEFAULT_AVATAR_PROMPT=A scene in which the anchorwoman is interacting with the audience, with a clean interior in the background.
EOF
```

## Step 4: Secure the File

```bash
chmod 600 .env
echo "✅ .env file created successfully"
```

## Step 5: Install Dependencies

```bash
pip install fastapi uvicorn[standard] websockets python-multipart pydantic pydantic-settings python-dotenv pyyaml openai elevenlabs pydub soundfile librosa aiofiles
```

This will take 1-2 minutes.

## Step 6: Verify Installation

```bash
python -c "import fastapi, openai, elevenlabs; print('✅ All dependencies installed')"
```

## Step 7: Start the Server

```bash
bash start_interactive_server.sh
```

## Expected Output

You should see:

```
==========================================
Starting Interactive LiveAvatar Server
==========================================
Checking dependencies...
Starting server...
[INFO] Initializing STT service...
[INFO] STT Service initialized with model: whisper-1
[INFO] Initializing LLM service...
[INFO] LLM Service initialized with model: gpt-3.5-turbo
[INFO] Initializing TTS service...
[INFO] TTS Service initialized with voice: 21m00Tcm4TlvDq8ikWAM
[INFO] Initializing Avatar service (this may take a while)...
[INFO] Creating WanS2V pipeline...
[INFO] Loading LoRA weights from Quark-Vision/Live-Avatar...
[INFO] Applying FP8 quantization...
[INFO] LiveAvatar pipeline initialized successfully!
[INFO] All services initialized successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 8: Find Your RunPod IP

In the SSH terminal:

```bash
curl ifconfig.me
```

This will show your public IP (e.g., `195.26.233.54`)

## Step 9: Access the Demo

Open your browser to:

```
http://195.26.233.54:8000
```

(Replace with your actual IP from step 8)

## Troubleshooting

### If server won't start:
```bash
# Check if files exist
ls -la | grep -E "(\.env|start_interactive|interactive_avatar)"

# Check .env is readable
cat .env | head -5
```

### If dependencies fail:
```bash
# Update pip first
pip install --upgrade pip
# Then retry step 5
```

### If port 8000 is blocked:
You may need to expose the port in RunPod dashboard:
1. Go to your pod settings
2. Add TCP port 8000
3. Use the exposed URL instead

---

**Once the server is running, let me know and I'll help you test it!**
