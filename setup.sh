#!/bin/bash
# =============================================================================
# Nectar AI — Full Setup
# Installs dependencies + downloads all required models
#
# Usage:
#   ./setup.sh              # full setup (deps + models)
#   ./setup.sh --deps-only  # only install pip dependencies
#   ./setup.sh --models-only # only download models
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="/workspace/frontend_demo/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
CONDA_ACTIVATE="source /opt/miniforge3/bin/activate"

# --- Parse flags ---
DO_DEPS=true
DO_MODELS=true
for arg in "$@"; do
    case $arg in
        --deps-only)  DO_MODELS=false ;;
        --models-only) DO_DEPS=false ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --deps-only    Only install Python dependencies"
            echo "  --models-only  Only download models"
            echo "  --help         Show this help"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Nectar AI — Full Setup               ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# =====================================================================
# Helper: download file if not already present
# =====================================================================
download_file() {
    local url=$1
    local folder=$2
    local name=$(basename "$url")
    local target="$MODELS_DIR/$folder"

    echo -e "${CYAN}  $name${NC}"

    mkdir -p "$target"

    if [ -f "$target/$name" ]; then
        echo -e "    ${GREEN}already exists, skipping${NC}"
        return
    fi

    wget -q --show-progress -c "$url" -P "$target"

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}done${NC}"
    else
        echo -e "    ${RED}FAILED${NC}"
    fi
}

# Download with custom output filename
download_file_as() {
    local url=$1
    local folder=$2
    local name=$3
    local target="$MODELS_DIR/$folder"

    echo -e "${CYAN}  $name${NC}"

    mkdir -p "$target"

    if [ -f "$target/$name" ]; then
        echo -e "    ${GREEN}already exists, skipping${NC}"
        return
    fi

    wget -q --show-progress -c "$url" -O "$target/$name"

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}done${NC}"
    else
        echo -e "    ${RED}FAILED${NC}"
    fi
}


# =====================================================================
# STEP 1: Install Python Dependencies
# =====================================================================
if [ "$DO_DEPS" = true ]; then
    echo -e "${YELLOW}[Step 1] Installing Python dependencies...${NC}"
    echo ""

    # --- 1a. ComfyUI + custom nodes (comfy conda env) ---
    echo -e "${CYAN}  [1a] ComfyUI core + custom nodes (comfy env)...${NC}"
    if [ -f /workspace/edgaras_IMAGE/requirements.txt ]; then
        $CONDA_ACTIVATE comfy && pip install -r /workspace/edgaras_IMAGE/requirements.txt 2>&1 | tail -5
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}  Warning: some ComfyUI deps had errors${NC}"
        fi
    else
        echo -e "${YELLOW}  /workspace/edgaras_IMAGE/requirements.txt not found, skipping${NC}"
    fi
    echo ""

    # --- 1b. Backend (FastAPI) deps (comfy conda env) ---
    echo -e "${CYAN}  [1b] Backend (FastAPI) dependencies (comfy env)...${NC}"
    $CONDA_ACTIVATE comfy && pip install -r "$SCRIPT_DIR/backend/requirements.txt" 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}  Warning: some backend deps had errors${NC}"
    fi
    echo ""

    # --- 1c. Frontend (Streamlit) deps (system python) ---
    echo -e "${CYAN}  [1c] Streamlit frontend (system python)...${NC}"
    pip install streamlit requests 2>&1 | tail -3
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}  Warning: streamlit install had errors${NC}"
    fi
    echo ""

    echo -e "${GREEN}  Dependencies installed.${NC}"
    echo ""
fi


# =====================================================================
# STEP 2: Download Models
# =====================================================================
if [ "$DO_MODELS" = true ]; then
    echo -e "${YELLOW}[Step 2] Downloading models...${NC}"
    echo ""

    if [ ! -d "$COMFYUI_DIR" ]; then
        echo -e "${RED}  ComfyUI directory not found at $COMFYUI_DIR${NC}"
        echo -e "${RED}  Please clone ComfyUI first.${NC}"
        exit 1
    fi

    # --- 2a. Core models (checkpoint, diffusion, text encoder, VAE) ---
    echo -e "${YELLOW}  [2a] Core models...${NC}"
    download_file "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" "checkpoints"
    download_file "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_nvfp4.safetensors" "diffusion_models"
    download_file "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors" "text_encoders"
    download_file "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" "vae"
    echo ""

    # --- 2b. IPAdapter FaceID models ---
    echo -e "${YELLOW}  [2b] IPAdapter FaceID models...${NC}"
    download_file "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin" "ipadapter"
    download_file "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin" "ipadapter"
    download_file "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors" "loras"
    echo ""

    # --- 2c. CLIP Vision model ---
    echo -e "${YELLOW}  [2c] CLIP Vision model...${NC}"
    download_file_as "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" "clip_vision" "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    echo ""

    # --- 2d. SAM3 (Segment Anything Model 3) ---
    echo -e "${YELLOW}  [2d] SAM3 model...${NC}"
    download_file "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt" "sam3"
    echo ""

    # --- 2e. InsightFace (buffalo_l) for face analysis ---
    echo -e "${YELLOW}  [2e] InsightFace buffalo_l models...${NC}"
    BUFFALO_DIR="$MODELS_DIR/insightface/models/buffalo_l"
    mkdir -p "$BUFFALO_DIR"
    BUFFALO_BASE="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l"
    for model_file in det_10g.onnx 1k3d68.onnx 2d106det.onnx genderage.onnx w600k_r50.onnx; do
        if [ -f "$BUFFALO_DIR/$model_file" ]; then
            echo -e "${CYAN}  $model_file${NC}"
            echo -e "    ${GREEN}already exists, skipping${NC}"
        else
            echo -e "${CYAN}  $model_file${NC}"
            wget -q --show-progress -c "$BUFFALO_BASE/$model_file" -O "$BUFFALO_DIR/$model_file"
        fi
    done
    echo ""

    # --- 2f. ControlNet ---
    echo -e "${YELLOW}  [2f] ControlNet model...${NC}"
    download_file "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin" "controlnet"
    echo ""

    # --- 2g. LoRA models (custom NSFW + pose) ---
    echo -e "${YELLOW}  [2g] LoRA models...${NC}"
    # These are custom LoRAs — check if they exist, warn if missing
    LORA_DIR="$MODELS_DIR/loras"
    REQUIRED_LORAS=(
        "PenisLora.safetensors"
        "multiConceptNSFW.safetensors"
        "Qwen4playNSFW.safetensors"
    )
    for lora in "${REQUIRED_LORAS[@]}"; do
        if [ -f "$LORA_DIR/$lora" ]; then
            echo -e "${CYAN}  $lora${NC}"
            echo -e "    ${GREEN}already exists${NC}"
        else
            echo -e "${CYAN}  $lora${NC}"
            echo -e "    ${RED}MISSING — upload manually to $LORA_DIR/${NC}"
        fi
    done
    echo ""

    echo -e "${GREEN}  Model downloads complete.${NC}"
    echo ""
fi


# =====================================================================
# STEP 3: Verify
# =====================================================================
echo -e "${YELLOW}[Verify] Checking critical files...${NC}"
echo ""

CHECKS=(
    "$MODELS_DIR/checkpoints/Qwen-Rapid-AIO-NSFW-v23.safetensors"
    "$MODELS_DIR/diffusion_models/z_image_turbo_nvfp4.safetensors"
    "$MODELS_DIR/text_encoders/qwen_3_4b_fp4_mixed.safetensors"
    "$MODELS_DIR/vae/ae.safetensors"
    "$MODELS_DIR/ipadapter/ip-adapter-faceid-plusv2_sd15.bin"
    "$MODELS_DIR/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    "$MODELS_DIR/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
    "$MODELS_DIR/loras/PenisLora.safetensors"
    "$MODELS_DIR/loras/multiConceptNSFW.safetensors"
    "$MODELS_DIR/insightface/models/buffalo_l/det_10g.onnx"
)

ALL_OK=true
for f in "${CHECKS[@]}"; do
    name=$(basename "$f")
    if [ -f "$f" ]; then
        echo -e "  ${GREEN}OK${NC}  $name"
    else
        echo -e "  ${RED}MISSING${NC}  $name"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Setup complete! All files present.   ${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}   Setup done. Some files are missing.  ${NC}"
    echo -e "${YELLOW}   Check the list above.                ${NC}"
    echo -e "${YELLOW}========================================${NC}"
fi
echo ""
echo -e "Next: run ${CYAN}./start.sh${NC} to start all services."
