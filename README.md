# Nectar AI — Multi-Modal Character Chat

A multi-modal AI character chat application that combines conversational AI with consistent character image generation. Users chat with AI characters and receive generated images that maintain the character's face identity across different poses, outfits, and backgrounds.

## How It Works

### Character Consistency via Chain-of-Edit

Image generation uses a **chain-of-edit** approach with the **Qwen edit AIO model**. Instead of generating images from scratch each time, the system makes sequential small edits to a reference image, preserving the character's identity throughout.

**Grok** acts as both the chatbot and orchestrator — it decomposes user requests into ordered editing steps and generates the prompt for each step. There are 3 editing step types:

- **Background** — change the scene/environment
- **Pose** — change the character's body position
- **Outfit** — change clothing or remove it

### LoRA Support

The Qwen edit model is plugged with specialized LoRAs:

| LoRA | Purpose |
|------|---------|
| **MCNL** (Multi Concept NSFW Lora) | NSFW content generation for female characters |
| **PenisLora** | NSFW content generation for male characters |
| **Denoising reduction LoRA** | Speeds up generation by reducing denoising steps |

Grok selects which LoRA is appropriate for each generation step. The backend overrides the LoRA switch, positive prompt, and image dimensions per step.

### Face Similarity Check

After each generation step, the ComfyUI workflow checks face similarity against the original reference image:

- Uses a custom ComfyUI node with a **CNN face detection model** + **face landmark detection**
- Similarity computed via **cosine similarity**
- Threshold: **40** (tuned empirically)
- If below threshold, the step reruns automatically (max **3 retries**)

### Why Not InstantID / IP-Adapter-FaceID?

These tools are designed for older models (SDXL, etc.) and do not work with the Qwen model family. Qwen Image edit has native LoRA support and is optimized for VRAM-limited GPUs.

### Alternatives Tried

- **Inpainting** for backgrounds — rejected because the model loses spatial and geometry consistency (cannot understand the whole image context)
- **ControlNet** for poses — works but limits the choices of poses, contradicting the goal of free-form user prompting

## Architecture


- **ComfyUI** — serves the Qwen edit model and workflows
- **FastAPI backend** — handles user requests, queues image generation jobs to ComfyUI
- **Grok** — chatbot character, orchestrator, prompt generator, and LoRA selector
- **Streamlit** — chat UI with character selection, persona editing, and image display

### AIO Workflow (`AIO.json`)

The primary image generation uses the `AIO.json` ComfyUI workflow (`USE_AIO_MODE=true`). This single workflow handles all editing steps (background, pose, outfit) in a unified pipeline:

1. **Load Checkpoint** — loads the `Qwen-Rapid-AIO-NSFW` model
2. **Prompt Encoding** — `TextEncodeQwenImageEditPlus` encodes the edit instruction
3. **LoRA Switch** — `ImpactSwitch` selects the appropriate LoRA per step (1 = none/SFW, 2 = PenisLora, 3 = MCNL NSFW)
4. **IPAdapter FaceID** — injects the character's face identity into the generation
5. **KSampler** — runs the diffusion edit (8 steps, euler_ancestral, denoise 0.9)
6. **Face Similarity** — compares the output face against the reference image and returns a score

The backend runs this workflow once per editing step. Each step's output image becomes the input for the next step, forming the chain-of-edit.

### Job-Based Async Image Generation

Image generation uses a job-based polling pattern to avoid Cloudflare tunnel timeouts:

1. `POST /chat/generate-image` — returns a `job_id` immediately
2. Client polls `GET /chat/job/{job_id}` every 3 seconds
3. Each poll is a fast dict lookup (~1ms), so the tunnel never times out

## Project Structure

```
├── start.sh                    # One-command launcher (tmux + cloudflared)
├── requirements.txt            # All Python deps (backend + Streamlit)
├── streamlit_app.py            # Streamlit chat frontend
├── backend/
│   ├── main.py                 # FastAPI app — all endpoints
│   ├── chat_service.py         # Grok chat with intent detection
│   ├── image_bridge_aio.py     # AIO mode — multi-step image generation
│   ├── image_bridge.py         # Non-AIO mode — pose + outfit workflows
│   ├── comfyui_client.py       # WebSocket client for ComfyUI
│   ├── character_profiles.py   # Character definitions and personas
│   ├── config.py               # Settings via pydantic-settings
│   ├── cost_logger.py          # Grok API cost tracking
│   ├── telegram_notify.py      # Telegram image delivery
│   ├── google_chat.py          # Google Chat webhook integration
│   ├── pose_generator.py       # Pose generation utilities
│   └── workflows/
│       ├── AIO.json            # All-in-one edit workflow
│       ├── Outfit.json         # Outfit change workflow
│       └── Pose.json           # Pose change workflow
├── frontend/                   # Next.js frontend (React + Tailwind)
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── components/
│   ├── package.json
│   └── ...
├── character/                  # Character reference images
│   ├── Luna.jpeg
│   ├── Sofia.jpeg
│   ├── Marco.jpg
│   ├── ...
│   └── uploads/                # User-uploaded custom character images
└── comfyui_custom_nodes/       # ComfyUI custom nodes (47 packages)
```

## Characters

15 pre-built characters with unique personas and reference images, plus 2 custom upload slots:

| Character | Gender | Personality | Occupation |
|-----------|--------|-------------|------------|
| Luna | Woman | Shy | Photographer |
| Sofia | Woman | Temptress | Model |
| Aria | Woman | Gentle | Yoga Instructor |
| Kai | Woman | Chill | Bartender |
| Elena | Woman | Mysterious | Dancer |
| Jade | Woman | Bold | DJ |
| Mia | Woman | Bubbly | Student |
| Nina | Woman | Sophisticated | Designer |
| Lena | Woman | Energetic | Fitness Trainer |
| Zara | Woman | Creative | Musician |
| Rosa | Woman | Passionate | Chef |
| Marco | Man | Confident | Artist |
| Justin | Man | Confident | Personal Trainer |
| Ivy | Man | Reserved | Librarian |
| Cleo | Man | Fiery | Dancer |
| Custom Woman | — | Configurable | — |
| Custom Man | — | Configurable | — |

Users can also upload their own reference images for custom characters.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/characters` | List all available characters |
| `GET` | `/characters/{id}/persona` | Get character persona details |
| `POST` | `/characters/upload-image` | Upload custom character reference image |
| `GET` | `/avatar/{id}` | Serve character avatar image |
| `POST` | `/chat/stream` | Send message, get text response (fast) |
| `POST` | `/chat/generate-image` | Submit image generation job (returns `job_id`) |
| `GET` | `/chat/job/{job_id}` | Poll image generation job status |
| `POST` | `/chat/job/{job_id}/send-telegram` | Send generated image to Telegram |
| `GET` | `/cost` | Get Grok API usage and cost stats |

## Setup

### Prerequisites

- Python 3.11+ (conda `comfy` environment recommended)
- ComfyUI with the Qwen edit model and custom nodes loaded
- xAI API key (for Grok)
- `tmux` and `cloudflared` installed

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This single file covers both backend (FastAPI) and frontend (Streamlit) dependencies.

### 2. Configure Environment

Create `backend/.env`:

```env
XAI_API_KEY=your-xai-api-key
COMFYUI_SERVER_ADDRESS=127.0.0.1:8188
USE_AIO_MODE=true
```

### 3. Start All Services

```bash
bash start.sh
```

This launches everything in a **tmux session** with 3 panes:

| Pane | Service | Port |
|------|---------|------|
| 0 | FastAPI backend | `8000` |
| 1 | Cloudflare tunnel | auto |
| 2 | Streamlit frontend | `8501` |

The script also:
- Starts ComfyUI automatically if not already running (via `edgaras_IMAGE/start.sh`)
- Waits for ComfyUI to be ready on port `8188`
- Captures the Cloudflare tunnel URL and writes it to `.streamlit/secrets.toml`

Use `--no-comfy` to skip starting ComfyUI if it's already running:

```bash
bash start.sh --no-comfy
```

### Managing the Session

```bash
tmux attach -t nectar      # Attach to the session
tmux kill-session -t nectar # Stop all services
```

### Next.js Frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Image Generation Model | Qwen Image Edit (AIO) |
| Image Serving | ComfyUI |
| Chat / Orchestration | Grok (`grok-4-fast-reasoning`) via xAI API |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit / Next.js (React + Tailwind) |
| Face Similarity | Custom CNN + cosine similarity (ComfyUI node) |
| Image Storage | Supabase Storage (optional) |
| Notifications | Telegram Bot, Google Chat Webhooks |
