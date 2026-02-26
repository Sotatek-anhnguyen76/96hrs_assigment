const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Character {
  id: string;
  name: string;
  description: string;
  has_ref_image: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  message: string;
  image_url: string | null;
  image_generating: boolean;
  image_context: string | null;
}

export async function fetchCharacters(): Promise<Character[]> {
  const res = await fetch(`${API_URL}/characters`);
  if (!res.ok) throw new Error("Failed to fetch characters");
  return res.json();
}

export async function sendMessage(
  characterId: string,
  message: string,
  conversationHistory: ChatMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      character_id: characterId,
      message,
      conversation_history: conversationHistory,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed");
  }
  return res.json();
}
