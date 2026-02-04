"use client";

import { useEffect, useMemo, useState } from "react";
import { Server, Copy, Check, RefreshCw, Play, Wrench } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useServerStatus } from "@/hooks/useServer";
import { useModels } from "@/hooks/useModels";
import { reloadAdapter } from "@/lib/api";
import { CommandCard } from "@/components/ui/command-card";
import { StatusPill } from "@/components/ui/status-pill";
import { useSettingsStore } from "@/stores/settings";

export default function ServePage() {
  const { data: serverStatus } = useServerStatus();
  const { data: models } = useModels();
  const { apiUrl } = useSettingsStore();
  const [port, setPort] = useState(8080);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedAdapter, setSelectedAdapter] = useState<string>("");
  const [reloadBase, setReloadBase] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [copied, setCopied] = useState(false);

  const baseModels = useMemo(() => models || [], [models]);
  const adapterModels = useMemo(() => (models || []).filter((m) => m.isAdapter), [models]);

  const serveCommand = `mlxsmith serve --model ${selectedModel || "<model-path>"} --port ${port}`;
  const apiBase = apiUrl.replace(/\/+$/, "");
  const chatCompletionsUrl = `${apiBase}/v1/chat/completions`;

  useEffect(() => {
    try {
      const parsed = new URL(apiBase);
      if (parsed.port) {
        setPort(Number(parsed.port));
      }
    } catch {
      // ignore invalid URLs
    }
  }, [apiBase]);

  useEffect(() => {
    if (!selectedModel && baseModels.length > 0) {
      setSelectedModel(baseModels[0].cliArgument);
    }
  }, [selectedModel, baseModels]);

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReload = async () => {
    if (!selectedAdapter) return;
    setReloading(true);
    try {
      await reloadAdapter(selectedAdapter, reloadBase);
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to reload adapter");
    } finally {
      setReloading(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Server className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Serve</h1>
            <p className="text-sm text-muted-foreground">OpenAI-compatible server controls</p>
          </div>
        </div>
        <StatusPill tone={serverStatus?.running ? "green" : "red"}>
          {serverStatus?.running ? "Running" : "Offline"}
        </StatusPill>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="mx-auto max-w-3xl space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Start Server
              </CardTitle>
              <CardDescription>Use the CLI to start the server on your machine.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Model</Label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="h-10 w-full rounded-lg border border-input/70 bg-background/70 px-3 text-sm backdrop-blur"
                  >
                    <option value="">Select a model</option>
                    {baseModels.map((model) => (
                      <option key={model.id} value={model.cliArgument}>
                        {model.displayName}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label>Port</Label>
                  <Input
                    type="number"
                    value={port}
                    onChange={(e) => setPort(Number(e.target.value))}
                    className="max-w-[200px]"
                  />
                </div>
              </div>
              <CommandCard
                title="Serve Command"
                description="Run from the project root to start the server."
                command={serveCommand}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wrench className="h-5 w-5" />
                Reload Adapter
              </CardTitle>
              <CardDescription>Hot-swap adapters without restarting the server.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Adapter</Label>
                  <select
                    value={selectedAdapter}
                    onChange={(e) => setSelectedAdapter(e.target.value)}
                    className="h-10 w-full rounded-lg border border-input/70 bg-background/70 px-3 text-sm backdrop-blur"
                  >
                    <option value="">Select an adapter</option>
                    {adapterModels.map((model) => (
                      <option key={model.id} value={model.cliArgument}>
                        {model.displayName}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end gap-2">
                  <Button variant="outline" onClick={handleReload} disabled={!selectedAdapter || reloading}>
                    <RefreshCw className={reloading ? "mr-2 h-4 w-4 animate-spin" : "mr-2 h-4 w-4"} />
                    Reload Adapter
                  </Button>
                  <Button
                    variant={reloadBase ? "glow" : "outline"}
                    onClick={() => setReloadBase(!reloadBase)}
                  >
                    {reloadBase ? "Reload Base: On" : "Reload Base: Off"}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>API Endpoints</CardTitle>
              <CardDescription>OpenAI-compatible endpoints available on the server.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Base URL</Label>
                <div className="flex gap-2">
                  <Input value={apiBase} readOnly />
                  <Button variant="outline" size="icon" onClick={() => copyToClipboard(apiBase)}>
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              <Separator />
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Chat Completions</Label>
                  <Badge variant="secondary">POST</Badge>
                </div>
                <div className="flex gap-2">
                  <Input value={chatCompletionsUrl} readOnly />
                  <Button variant="outline" size="icon" onClick={() => copyToClipboard(chatCompletionsUrl)}>
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <CommandCard
            title="Example cURL"
            description="Quick smoke test for chat completions."
            command={`curl ${chatCompletionsUrl} \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 256}'`}
          />
        </div>
      </ScrollArea>
    </div>
  );
}
