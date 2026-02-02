"use client";

import { useEffect } from "react";
import { MessageSquare } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Message } from "@/components/chat/message";
import { ChatInput } from "@/components/chat/chat-input";
import { ChatSidebar } from "@/components/chat/chat-sidebar";
import { useChatStore } from "@/stores/chat";
import { useSettingsStore } from "@/stores/settings";
import { useChat } from "@/hooks/useChat";

export default function ChatPage() {
  const { sessions, currentSessionId, createSession } = useChatStore();
  const { defaultModel, defaultSystemPrompt, defaultTemperature, defaultTopP, defaultMaxTokens } = useSettingsStore();
  const { sendMessage, isStreaming } = useChat();

  // Create initial session if none exists
  useEffect(() => {
    if (sessions.length === 0) {
      createSession(defaultModel);
    }
  }, [sessions.length, defaultModel, createSession]);

  const currentSession = sessions.find((s) => s.id === currentSessionId);

  const handleSendMessage = async (content: string) => {
    if (!currentSession) return;

    const messages = [
      { role: "system" as const, content: defaultSystemPrompt },
      ...currentSession.messages,
      { role: "user" as const, content },
    ];

    await sendMessage(currentSession.id, content, {
      model: currentSession.model || defaultModel,
      messages,
      max_tokens: defaultMaxTokens,
      temperature: defaultTemperature,
      top_p: defaultTopP,
      stream: true,
    });
  };

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <ChatSidebar />

      {/* Main Chat Area */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="flex h-16 items-center border-b px-6">
          <MessageSquare className="mr-2 h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Chat</h1>
            {currentSession && (
              <p className="text-sm text-muted-foreground">
                {currentSession.model || defaultModel}
              </p>
            )}
          </div>
        </header>

        {/* Messages */}
        <ScrollArea className="flex-1 px-6">
          {currentSession ? (
            <div className="py-4">
              {currentSession.messages.length === 0 ? (
                <div className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                  <MessageSquare className="mb-4 h-12 w-12 opacity-20" />
                  <p>Start a conversation</p>
                </div>
              ) : (
                currentSession.messages.map((message, index) => (
                  <Message key={index} message={message} />
                ))
              )}
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center text-muted-foreground">
              <p>Create a new chat to start</p>
            </div>
          )}
        </ScrollArea>

        {/* Input */}
        <ChatInput
          onSend={handleSendMessage}
          isLoading={isStreaming}
          disabled={!currentSession}
        />
      </div>
    </div>
  );
}
