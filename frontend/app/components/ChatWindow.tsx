"use client";

import { useEffect, useRef } from "react";

export interface DisplayMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  imageUrl?: string | null;
  imageContext?: string | null;
}

interface Props {
  messages: DisplayMessage[];
  characterName: string;
  isLoading: boolean;
}

export default function ChatWindow({
  messages,
  characterName,
  isLoading,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.length === 0 && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center text-[var(--muted)]">
            <p className="text-xl mb-2">Start chatting with {characterName}</p>
            <p className="text-sm">
              Try asking &quot;What are you wearing?&quot; or &quot;Send me a
              selfie&quot;
            </p>
          </div>
        </div>
      )}

      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-[75%] rounded-2xl px-4 py-3 ${
              msg.role === "user"
                ? "bg-[var(--user-bubble)] text-[var(--foreground)]"
                : "bg-[var(--ai-bubble)] text-[var(--foreground)] border border-[var(--border)]"
            }`}
          >
            {msg.role === "assistant" && (
              <div className="text-xs text-[var(--accent)] font-semibold mb-1">
                {characterName}
              </div>
            )}
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {msg.content}
            </p>

            {msg.imageUrl && (
              <div className="mt-3 rounded-xl overflow-hidden border border-[var(--border)]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={msg.imageUrl}
                  alt={msg.imageContext || "Character photo"}
                  className="w-full max-w-sm rounded-xl"
                  loading="lazy"
                />
              </div>
            )}

            {msg.imageUrl === undefined && msg.imageContext && (
              <div className="mt-3 w-64 h-80 rounded-xl image-loading flex items-center justify-center">
                <span className="text-xs text-[var(--muted)]">
                  Generating photo...
                </span>
              </div>
            )}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-[var(--ai-bubble)] border border-[var(--border)] rounded-2xl px-4 py-3">
            <div className="text-xs text-[var(--accent)] font-semibold mb-1">
              {characterName}
            </div>
            <div className="flex gap-1">
              <div className="typing-dot w-2 h-2 bg-[var(--muted)] rounded-full" />
              <div className="typing-dot w-2 h-2 bg-[var(--muted)] rounded-full" />
              <div className="typing-dot w-2 h-2 bg-[var(--muted)] rounded-full" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
