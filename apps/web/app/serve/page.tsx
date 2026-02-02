"use client";

import { useState } from "react";
import { Server, Power, ExternalLink, Copy, Check } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useServerStatus, useToggleServer } from "@/hooks/useServer";
import { useServerStore } from "@/stores/server";
import { cn } from "@/lib/utils";

export default function ServePage() {
  const { data: serverStatus } = useServerStatus();
  const { toggle, isLoading } = useToggleServer();
  const { isRunning, port } = useServerStore();
  const [copied, setCopied] = useState(false);

  const apiUrl = `http://localhost:${port}`;
  const chatCompletionsUrl = `${apiUrl}/v1/chat/completions`;

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-16 items-center justify-between border-b px-6">
        <div className="flex items-center gap-4">
          <Server className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Serve</h1>
            <p className="text-sm text-muted-foreground">
              API server management
            </p>
          </div>
        </div>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1 p-6">
        <div className="mx-auto max-w-2xl space-y-6">
          {/* Status Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Power className="h-5 w-5" />
                Server Status
              </CardTitle>
              <CardDescription>
                Toggle the API server on or off
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between rounded-lg border p-4">
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      "h-3 w-3 rounded-full",
                      isRunning ? "bg-green-500" : "bg-red-500"
                    )}
                  />
                  <div>
                    <p className="font-medium">
                      {isRunning ? "Running" : "Stopped"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isRunning
                        ? `Server is listening on port ${port}`
                        : "Server is currently offline"}
                    </p>
                  </div>
                </div>
                <Switch
                  checked={isRunning}
                  onCheckedChange={toggle}
                  disabled={isLoading}
                />
              </div>

              {/* Port Configuration */}
              <div className="space-y-2">
                <Label>Port</Label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    value={port}
                    disabled={isRunning || isLoading}
                    className="max-w-[200px]"
                  />
                  <Button variant="outline" disabled={isRunning || isLoading}>
                    Update
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* API Endpoints */}
          <Card>
            <CardHeader>
              <CardTitle>API Endpoints</CardTitle>
              <CardDescription>
                OpenAI-compatible API endpoints
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Base URL */}
              <div className="space-y-2">
                <Label>Base URL</Label>
                <div className="flex gap-2">
                  <Input value={apiUrl} readOnly />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(apiUrl)}
                  >
                    {copied ? (
                      <Check className="h-4 w-4" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <Separator />

              {/* Chat Completions */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Chat Completions</Label>
                  <Badge variant="secondary">POST</Badge>
                </div>
                <div className="flex gap-2">
                  <Input value={chatCompletionsUrl} readOnly />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(chatCompletionsUrl)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Example cURL */}
              <div className="space-y-2">
                <Label>Example cURL Request</Label>
                <pre className="relative overflow-x-auto rounded-lg bg-muted p-4 text-xs">
                  <code>{`curl ${chatCompletionsUrl} \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'`}</code>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-2 top-2"
                    onClick={() =>
                      copyToClipboard(
                        `curl ${chatCompletionsUrl} -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 256}'`
                      )
                    }
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                </pre>
              </div>
            </CardContent>
          </Card>

          {/* OpenAI SDK Config */}
          <Card>
            <CardHeader>
              <CardTitle>OpenAI SDK Configuration</CardTitle>
              <CardDescription>
                Use with the OpenAI Python or JavaScript SDK
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="relative overflow-x-auto rounded-lg bg-muted p-4 text-xs">
                <code>{`from openai import OpenAI

client = OpenAI(
    base_url="${apiUrl}",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)`}</code>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-2 top-2"
                  onClick={() =>
                    copyToClipboard(`from openai import OpenAI

client = OpenAI(
    base_url="${apiUrl}",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)`)
                  }
                >
                  <Copy className="h-3 w-3" />
                </Button>
              </pre>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  );
}
