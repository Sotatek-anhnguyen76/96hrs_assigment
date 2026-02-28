#!/bin/bash
# =============================================================================
# Start Nectar AI — Backend (FastAPI) + Frontend (Streamlit) + Cloudflare Tunnel
# Uses tmux with 3 panes: backend | frontend | tunnel
# =============================================================================

SESSION="nectar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
CONDA_BIN="/opt/miniforge3/bin/conda"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Parse flags ---
SKIP_COMFY=false
for arg in "$@"; do
    case $arg in
        --no-comfy) SKIP_COMFY=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-comfy    Skip starting ComfyUI (if already running)"
            echo "  --help        Show this help"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Nectar AI — Starting Services        ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# --- Kill existing session ---
tmux kill-session -t "$SESSION" 2>/dev/null

# --- Start ComfyUI if needed ---
if [ "$SKIP_COMFY" = false ]; then
    echo -e "${YELLOW}[1/4] Starting ComfyUI...${NC}"
    # Check if edgaras start.sh exists and ComfyUI is not already running
    if ! curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        if [ -f /workspace/edgaras_IMAGE/start.sh ]; then
            echo -e "${CYAN}  Running edgaras_IMAGE/start.sh in background...${NC}"
            bash /workspace/edgaras_IMAGE/start.sh &
            disown
            # Wait for ComfyUI to be ready
            echo -ne "${CYAN}  Waiting for ComfyUI (port 8188)...${NC}"
            for i in $(seq 1 60); do
                if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
                    echo -e " ${GREEN}ready!${NC}"
                    break
                fi
                echo -n "."
                sleep 2
            done
        else
            echo -e "${YELLOW}  edgaras_IMAGE/start.sh not found, skipping ComfyUI${NC}"
        fi
    else
        echo -e "${GREEN}  ComfyUI already running on port 8188${NC}"
    fi
else
    echo -e "${YELLOW}[1/4] Skipping ComfyUI (--no-comfy)${NC}"
fi
echo ""

# --- Create tmux session ---
echo -e "${YELLOW}[2/4] Starting Backend (FastAPI on port 8000)...${NC}"
tmux new-session -d -s "$SESSION" -n "services"

# Pane 0: Backend
tmux send-keys -t "$SESSION:services.0" \
    "cd $BACKEND_DIR && source /opt/miniforge3/bin/activate comfy && python main.py 2>&1 | tee $BACKEND_DIR/backend.log" Enter

# --- Pane 1: Cloudflare Tunnel ---
echo -e "${YELLOW}[3/4] Starting Cloudflare Tunnel...${NC}"
tmux split-window -h -t "$SESSION:services"
tmux send-keys -t "$SESSION:services.1" \
    "sleep 3 && cloudflared tunnel --url http://localhost:8000 2>&1 | tee $SCRIPT_DIR/tunnel.log" Enter

# --- Pane 2: Streamlit Frontend ---
echo -e "${YELLOW}[4/4] Starting Streamlit Frontend (port 8501)...${NC}"
tmux split-window -v -t "$SESSION:services.0"
tmux send-keys -t "$SESSION:services.2" \
    "sleep 5 && cd $SCRIPT_DIR && streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 2>&1 | tee $SCRIPT_DIR/streamlit.log" Enter

# --- Layout ---
tmux select-layout -t "$SESSION:services" main-vertical

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   All services starting!               ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Backend:   ${CYAN}http://localhost:8000${NC}"
echo -e "  Streamlit: ${CYAN}http://localhost:8501${NC}"
echo -e "  ComfyUI:   ${CYAN}http://localhost:8188${NC}"
echo -e "  Tunnel:    ${CYAN}check tunnel.log for public URL${NC}"
echo ""
echo -e "  tmux attach: ${YELLOW}tmux attach -t $SESSION${NC}"
echo -e "  Stop all:    ${YELLOW}tmux kill-session -t $SESSION${NC}"
echo ""

# --- Wait for tunnel URL and update secrets.toml ---
(
    sleep 8
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$SCRIPT_DIR/tunnel.log" 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo -e "\n${GREEN}Tunnel URL: $TUNNEL_URL${NC}"
        # Update secrets.toml for Streamlit
        mkdir -p "$SCRIPT_DIR/.streamlit"
        cat > "$SCRIPT_DIR/.streamlit/secrets.toml" <<EOF
# Backend API URL — auto-updated by start.sh
API_URL = "$TUNNEL_URL"
EOF
        echo -e "${GREEN}Updated .streamlit/secrets.toml with tunnel URL${NC}"
    fi
) &

# Attach to tmux
tmux attach -t "$SESSION"
