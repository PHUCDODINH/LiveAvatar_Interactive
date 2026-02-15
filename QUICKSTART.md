# Interactive LiveAvatar - Quick Start Guide

## 🚀 Quick Setup (On RunPod)

### 1. Install Dependencies

```bash
cd /root/LiveAvatar  # or wherever you cloned the repo
pip install -r requirements_interactive.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

**Required API Keys:**
- `OPENAI_API_KEY` - From https://platform.openai.com/api-keys
- `ELEVENLABS_API_KEY` - From https://elevenlabs.io/app/settings/api-keys

### 3. Start the Server

```bash
bash start_interactive_server.sh
```

### 4. Access the Interface

**If accessing from same network:**
```
http://<runpod-ip>:8000
```

**If using RunPod's exposed TCP port:**
1. Go to RunPod dashboard
2. Click on your pod → "Connect" → "HTTP Service"
3. Use the provided URL

---

## 📋 System Requirements

- ✅ RTX 6000 ADA (48GB VRAM) - Confirmed compatible
- ✅ CUDA 12.4+
- ✅ Python 3.10
- ✅ 100GB+ disk space
- ✅ Internet connection (for API calls)

---

## 🎮 Usage

### Via Microphone (Recommended)
1. Click and **hold** the microphone button
2. Speak your message
3. Release the button
4. Wait for the avatar to respond

### Via Text Input
1. Type your message in the text box
2. Click send or press Enter
3. Wait for the avatar to respond

---

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Performance (15-22 FPS target)
LIVEAVATAR_SIZE=640*360          # Lower = faster
LIVEAVATAR_INFER_FRAMES=32       # Lower = faster but shorter clips
LIVEAVATAR_SAMPLE_STEPS=2        # Lower = faster but lower quality
ENABLE_FP8=true                  # Memory efficient
ENABLE_COMPILE=true              # Faster after warmup

# Models
OPENAI_MODEL=gpt-3.5-turbo       # or gpt-4o for better quality
ELEVENLABS_VOICE_ID=...          # Change voice

# Avatar
DEFAULT_AVATAR_IMAGE=examples/anchor.jpg  # Your reference image
```

---

## 🔧 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>
```

### Out of Memory (OOM)
```bash
# Reduce settings in .env
LIVEAVATAR_SIZE=512*288
LIVEAVATAR_INFER_FRAMES=24
```

### Slow generation
- First generation will be slow (model compilation)
- Subsequent generations should be faster
- Check GPU utilization: `nvidia-smi`

### API Errors
- Verify API keys are correct in `.env`
- Check API quotas/credits
- Check internet connection

---

## 📊 Expected Performance

| Setting | FPS | Quality | Latency |
|---------|-----|---------|---------|
| Optimized (Default) | 15-22 | Good | ~2s |
| Balanced | 8-12 | Very Good | ~3s |
| Quality | 4-6 | Excellent | ~5s |

**Total Response Time Breakdown:**
- STT (Whisper): ~500ms
- LLM (GPT): ~1-2s
- TTS (ElevenLabs): ~1s
- Avatar Generation: ~2-4s
- **Total**: ~4.5-7.5s per interaction

---

## 🎯 Next Steps

1. **Test Basic Flow**: Start server and test with "Hello, how are you?"
2. **Customize Avatar**: Replace `DEFAULT_AVATAR_IMAGE` with your own
3. **Tune Performance**: Adjust settings based on your needs
4. **Add Personality**: Update LLM system prompt in `services/llm_service.py`

---

## 📝 API Costs (Estimate)

Per interaction (~100 words):
- OpenAI STT: $0.006
- OpenAI GPT-3.5: $0.0001
- ElevenLabs TTS: $0.003
- **Total**: ~$0.01 per interaction

100 interactions/day ≈ $1/day in API costs
