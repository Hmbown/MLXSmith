"use client";

import { useState } from "react";
import { Plus, Trash2, MessageSquare, Settings2, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useChatStore } from "@/stores/chat";
import { useSettingsStore } from "@/stores/settings";
import { cn } from "@/lib/utils";
import { useModels } from "@/hooks/useModels";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatusPill } from "@/components/ui/status-pill";

export function ChatSidebar() {
  const { sessions, currentSessionId, createSession, deleteSession, setCurrentSession } = useChatStore();
  const { defaultModel, setDefaultModel, defaultSystemPrompt, setDefaultSystemPrompt, defaultTemperature, setDefaultTemperature, defaultTopP, setDefaultTopP, defaultMaxTokens, setDefaultMaxTokens } = useSettingsStore();
  const { data: models } = useModels();
  const [showSettings, setShowSettings] = useState(true);

  return (
    <div className="flex h-full w-80 flex-col border-r border-border/60 bg-card/40 backdrop-blur">
      {/* New Chat Button */}
      <div className="p-4">
        <Button
          onClick={() => createSession(defaultModel)}
          className="w-full"
          variant="glass"
        >
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>

      <Separator />

      {/* Chat History */}
      <ScrollArea className="flex-1 p-2">
        <div className="space-y-1">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => setCurrentSession(session.id)}
              className={cn(
                "group flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm transition-colors",
                currentSessionId === session.id
                  ? "bg-accent/70 text-accent-foreground"
                  : "hover:bg-accent/40"
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate">{session.title}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(session.id);
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </button>
          ))}
        </div>
      </ScrollArea>

      <Separator />

      {/* Settings */}
      <div className="p-4">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="mb-3 flex w-full items-center justify-between text-sm font-medium"
        >
          <span className="flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4" />
            Session Defaults
          </span>
          <StatusPill tone="blue" dot={false}>
            Live
          </StatusPill>
        </button>

        {showSettings && (
          <div className="space-y-4">
            {/* Model Select */}
            <div className="space-y-2">
              <Label className="text-xs">Model</Label>
              <Select value={defaultModel} onValueChange={setDefaultModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models?.map((model) => (
                    <SelectItem key={model.id} value={model.fullName}>
                      {model.displayName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Temperature</Label>
                <span className="text-xs text-muted-foreground">{defaultTemperature}</span>
              </div>
              <Slider
                value={[defaultTemperature]}
                onValueChange={([v]) => setDefaultTemperature(v)}
                min={0}
                max={2}
                step={0.1}
              />
            </div>

            {/* Top P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Top P</Label>
                <span className="text-xs text-muted-foreground">{defaultTopP}</span>
              </div>
              <Slider
                value={[defaultTopP]}
                onValueChange={([v]) => setDefaultTopP(v)}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            {/* Max Tokens */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Max Tokens</Label>
                <span className="text-xs text-muted-foreground">{defaultMaxTokens}</span>
              </div>
              <Slider
                value={[defaultMaxTokens]}
                onValueChange={([v]) => setDefaultMaxTokens(v)}
                min={64}
                max={4096}
                step={64}
              />
            </div>

            {/* System Prompt */}
            <div className="space-y-2">
              <Label className="text-xs">System Prompt</Label>
              <textarea
                value={defaultSystemPrompt}
                onChange={(e) => setDefaultSystemPrompt(e.target.value)}
                className="min-h-[80px] w-full rounded-lg border border-input/70 bg-background/70 px-3 py-2 text-xs backdrop-blur"
                placeholder="Enter system prompt..."
              />
            </div>

            {/* Stream Toggle */}
            <div className="flex items-center justify-between">
              <Label className="text-xs">Stream Response</Label>
              <Switch defaultChecked />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
