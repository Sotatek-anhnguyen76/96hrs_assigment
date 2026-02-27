"""
Nectar AI — Multi-Modal Character Chat
Streamlit frontend that connects to the FastAPI backend on Vast.ai.

Uses split flow: /chat/stream for fast text, /chat/generate-image for async image gen.
"""
import streamlit as st
import requests

# Backend API URL — set in Streamlit Cloud secrets (Settings > Secrets)
# Format: API_URL = "https://your-tunnel-url.trycloudflare.com"
import os
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.environ.get("API_URL", "https://rental-presenting-compatibility-signing.trycloudflare.com")

st.set_page_config(page_title="Nectar AI Chat", page_icon="💬", layout="wide")

# --- State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "selected_character" not in st.session_state:
    st.session_state.selected_character = None


# --- API calls ---
@st.cache_data(ttl=10)
def fetch_characters():
    try:
        r = requests.get(f"{API_URL}/characters", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Cannot reach backend: {e}")
        return []


def send_text_only(character_id: str, message: str, history: list) -> dict:
    """Fast text response — no image generation."""
    r = requests.post(
        f"{API_URL}/chat/stream",
        json={
            "character_id": character_id,
            "message": message,
            "conversation_history": history,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def generate_image(character_id: str, image_context: str, pose_description=None, outfit_description=None) -> dict:
    """Trigger image generation separately."""
    r = requests.post(
        f"{API_URL}/chat/generate-image",
        json={
            "character_id": character_id,
            "image_context": image_context,
            "pose_description": pose_description,
            "outfit_description": outfit_description,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def fetch_cost():
    try:
        r = requests.get(f"{API_URL}/cost", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def full_image_url(url):
    """Convert relative /images/... path to full public URL."""
    if url and url.startswith("/"):
        return f"{API_URL}{url}"
    return url


# --- Sidebar: Character Select + Cost ---
with st.sidebar:
    st.markdown("## Nectar AI")
    st.caption("Multi-Modal Character Chat")
    st.divider()

    characters = fetch_characters()

    for char in characters:
        selected = st.session_state.selected_character
        is_selected = selected and selected["id"] == char["id"]

        if st.button(
            f"**{char['name']}**",
            key=f"char_{char['id']}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            if not is_selected:
                st.session_state.selected_character = char
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

    st.divider()

    # Cost tracker
    cost = fetch_cost()
    if cost:
        st.markdown("**API Cost**")
        col1, col2 = st.columns(2)
        col1.metric("Calls", cost.get("total_calls", 0))
        col2.metric("Cost", f"${cost.get('total_cost_usd', 0):.4f}")
        st.caption(
            f"In: {cost.get('total_input_tokens', 0):,} | "
            f"Out: {cost.get('total_output_tokens', 0):,} tokens"
        )


# --- Main Chat Area ---
char = st.session_state.selected_character

if not char:
    st.title("Welcome to Nectar AI")
    st.write("Select a character from the sidebar to start chatting.")
else:
    st.title(char["name"])

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("image_url"):
                st.image(full_image_url(msg["image_url"]), caption=msg.get("image_context", ""), width=400)

    # Chat input
    if prompt := st.chat_input(f"Message {char['name']}..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Step 1: Get text response fast
        with st.chat_message("assistant"):
            with st.spinner(f"{char['name']} is thinking..."):
                try:
                    text_resp = send_text_only(
                        char["id"],
                        prompt,
                        st.session_state.conversation_history,
                    )

                    st.write(text_resp["message"])

                    # Save text to state
                    msg_data = {
                        "role": "assistant",
                        "content": text_resp["message"],
                        "image_url": None,
                        "image_context": text_resp.get("image_context"),
                    }

                    st.session_state.conversation_history.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_resp["message"]},
                    ])

                    # Step 2: If image requested, generate it
                    if text_resp.get("send_image") and text_resp.get("image_context"):
                        with st.spinner("Generating image..."):
                            img_resp = generate_image(
                                char["id"],
                                text_resp.get("image_context", ""),
                            )

                            if img_resp.get("status") == "succeeded" and img_resp.get("image_url"):
                                img_url = full_image_url(img_resp["image_url"])
                                st.image(img_url, caption=text_resp.get("image_context", ""), width=400)
                                msg_data["image_url"] = img_resp["image_url"]
                            else:
                                st.warning(f"Image generation failed: {img_resp.get('error', 'unknown')}")

                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    st.error(f"Error: {e}")
