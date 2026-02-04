"use client";

import Link from "next/link";
import { LayoutDashboard, Sparkles, Cpu, Layers, Server, ArrowUpRight, Activity } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusPill } from "@/components/ui/status-pill";
import { IconBadge } from "@/components/ui/icon-badge";
import { useModels } from "@/hooks/useModels";
import { useServerStatus } from "@/hooks/useServer";
import { useRLMState } from "@/hooks/useRLM";
import { formatBytes, formatDate } from "@/lib/utils";

export default function DashboardPage() {
  const { data: models, isError: modelsError } = useModels();
  const { data: serverStatus } = useServerStatus();
  const { data: rlmState } = useRLMState();

  const localModels = models?.filter((m) => !m.isAdapter) || [];
  const adapters = models?.filter((m) => m.isAdapter) || [];
  const totalSizeBytes = models?.reduce((acc, m) => acc + (m.size_bytes || 0), 0) || 0;
  const recent = [...(models || [])]
    .filter((m) => m.downloaded_at)
    .sort((a, b) => (b.downloaded_at || 0) - (a.downloaded_at || 0))
    .slice(0, 4);

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <LayoutDashboard className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Dashboard</h1>
            <p className="text-sm text-muted-foreground">System status and recent activity</p>
          </div>
        </div>
        <StatusPill tone={serverStatus?.running ? "green" : "red"}>
          {serverStatus?.running ? "API Online" : "API Offline"}
        </StatusPill>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card>
              <CardHeader>
                <CardDescription>Local Models</CardDescription>
                <CardTitle className="text-2xl">{localModels.length}</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Cached MLX/HF entries</span>
                <IconBadge tone="blue">
                  <Cpu className="h-4 w-4" />
                </IconBadge>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Adapters</CardDescription>
                <CardTitle className="text-2xl">{adapters.length}</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-between text-sm text-muted-foreground">
                <span>LoRA and fine-tune outputs</span>
                <IconBadge tone="purple">
                  <Layers className="h-4 w-4" />
                </IconBadge>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>Cache Size</CardDescription>
                <CardTitle className="text-2xl">{formatBytes(totalSizeBytes)}</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Models directory usage</span>
                <IconBadge tone="yellow">
                  <Sparkles className="h-4 w-4" />
                </IconBadge>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardDescription>RLM Loop</CardDescription>
                <CardTitle className="text-2xl capitalize">{rlmState?.status || "idle"}</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-between text-sm text-muted-foreground">
                <span>
                  {rlmState?.iteration ? `Iteration ${rlmState.iteration}` : "No active loop"}
                </span>
                <IconBadge tone="green">
                  <Activity className="h-4 w-4" />
                </IconBadge>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Jump into common workflows</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 md:grid-cols-2">
                <Button asChild variant="glass" className="justify-between">
                  <Link href="/discover">
                    Browse models
                    <ArrowUpRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="glass" className="justify-between">
                  <Link href="/chat">
                    Start a chat
                    <ArrowUpRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="glass" className="justify-between">
                  <Link href="/train">
                    Configure training
                    <ArrowUpRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="glass" className="justify-between">
                  <Link href="/serve">
                    Serve a model
                    <ArrowUpRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Downloads</CardTitle>
                <CardDescription>Newest models in cache</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {modelsError && (
                  <div className="text-muted-foreground">
                    Could not reach the API. Check your API URL in Settings.
                  </div>
                )}
                {!modelsError && recent.length === 0 && (
                  <div className="text-muted-foreground">No models found yet.</div>
                )}
                {recent.map((model) => (
                  <div key={model.id} className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{model.displayName}</div>
                      <div className="text-xs text-muted-foreground">{model.fullName}</div>
                    </div>
                    <div className="text-right text-xs text-muted-foreground">
                      <div>{model.sizeDisplay}</div>
                      {model.downloaded_at && <div>{formatDate(model.downloaded_at * 1000)}</div>}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Serve Status</CardTitle>
              <CardDescription>OpenAI-compatible server availability</CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <IconBadge tone={serverStatus?.running ? "green" : "red"}>
                  <Server className="h-4 w-4" />
                </IconBadge>
                <div>
                  <div className="font-medium">
                    {serverStatus?.running ? "Server reachable" : "Server offline"}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {serverStatus?.running
                      ? "Use the Serve tab to reload adapters and copy endpoints."
                      : "Start the server with the CLI or Serve tab guidance."}
                  </div>
                </div>
              </div>
              <Button asChild variant="outline">
                <Link href="/serve">Open Serve</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  );
}
