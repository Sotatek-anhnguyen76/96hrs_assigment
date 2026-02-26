"use client";

import { useState, useEffect, useCallback } from "react";
import CharacterSelect from "./components/CharacterSelect";
import ChatWindow, { DisplayMessage } from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import {
  Character,
  ChatMessage,
  fetchCharacters,
  sendMessage,
} from "@/lib/api";

export default function Home() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(
    null
  );
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>(
    []
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCharacters()
      .then(setCharacters)
      .catch((e) => setError(e.message));
  }, []);

  const handleSelectCharacter = useCallback((c: Character) => {
    setSelectedCharacter(c);
    setMessages([]);
    setConversationHistory([]);
    setError(null);
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      if (!selectedCharacter) return;
      setError(null);

      // Add user message
      const userMsg: DisplayMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: text,
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const response = await sendMessage(
          selectedCharacter.id,
          text,
          conversationHistory
        );

        // Add assistant message
        const assistantMsg: DisplayMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: response.message,
          imageUrl: response.image_url,
          imageContext: response.image_context,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        // Update conversation history (text only, for context)
        setConversationHistory((prev) => [
          ...prev,
          { role: "user", content: text },
          { role: "assistant", content: response.message },
        ]);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
      } finally {
        setIsLoading(false);
      }
    },
    [selectedCharacter, conversationHistory]
  );

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <div className="w-72 border-r border-[var(--border)] flex flex-col bg-[var(--background)]">
        <div className="p-4 border-b border-[var(--border)]">
          <h1 className="text-xl font-bold text-[var(--accent)]">Nectar AI</h1>
          <p className="text-xs text-[var(--muted)] mt-1">
            Multi-Modal Character Chat
          </p>
        </div>
        <div className="flex-1 overflow-y-auto">
          <CharacterSelect
            characters={characters}
            selected={selectedCharacter}
            onSelect={handleSelectCharacter}
          />
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {selectedCharacter ? (
          <>
            {/* Header */}
            <div className="border-b border-[var(--border)] px-6 py-4 flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-[var(--accent)] flex items-center justify-center text-white font-bold text-sm">
                {selectedCharacter.name[0]}
              </div>
              <div>
                <div className="font-semibold text-[var(--foreground)]">
                  {selectedCharacter.name}
                </div>
                <div className="text-xs text-[var(--muted)]">Online</div>
              </div>
            </div>

            {/* Messages */}
            <ChatWindow
              messages={messages}
              characterName={selectedCharacter.name}
              isLoading={isLoading}
            />

            {/* Error banner */}
            {error && (
              <div className="px-4 py-2 bg-red-500/10 border-t border-red-500/20 text-red-400 text-sm text-center">
                {error}
              </div>
            )}

            {/* Input */}
            <ChatInput onSend={handleSend} disabled={isLoading} />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-[var(--muted)]">
              <p className="text-2xl mb-2">Welcome to Nectar AI</p>
              <p className="text-sm">
                Select a character from the sidebar to start chatting
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
