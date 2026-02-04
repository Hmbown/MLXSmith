"use client";

import { useMemo, useState } from "react";
import { Layers, RefreshCw, ArrowLeftRight, Wand2, MessageSquare, Server, Trash2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { StatusPill } from "@/components/ui/status-pill";
import { IconBadge } from "@/components/ui/icon-badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useModels, useDeleteModel } from "@/hooks/useModels";
import { reloadAdapter } from "@/lib/api";
import { useRouter } from "next/navigation";
import { useSettingsStore } from "@/stores/settings";
import type { LocalModel } from "@/lib/models";

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function asString(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number") return value.toString();
  return undefined;
}

function adapterSummary(model: LocalModel) {
  const metadata = model.metadata || {};
  return {
    trainingMode: asString(metadata["training_mode"]) || "LoRA",
    iterations: asNumber(metadata["iterations"]),
    bestScore: asNumber(metadata["best_score"]),
    loraRank: asNumber(metadata["lora_rank"]),
    loraAlpha: asNumber(metadata["lora_alpha"]),
  };
}

export default function AdaptersPage() {
  const { data: models, isLoading, refetch } = useModels();
  const deleteMutation = useDeleteModel();
  const router = useRouter();
  const { setDefaultModel } = useSettingsStore();
  const [selectedBase, setSelectedBase] = useState<string>("all");
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareA, setCompareA] = useState<string>("");
  const [compareB, setCompareB] = useState<string>("");
  const [reloading, setReloading] = useState<string | null>(null);

  const adapters = useMemo(() => (models || []).filter((m) => m.isAdapter), [models]);
  const baseModels = useMemo(() => {
    const list = adapters.map((a) => a.baseModel).filter(Boolean) as string[];
    return Array.from(new Set(list)).sort();
  }, [adapters]);

  const filtered = selectedBase === "all"
    ? adapters
    : adapters.filter((a) => a.baseModel === selectedBase);

  const handleChat = (adapter: LocalModel) => {
    setDefaultModel(adapter.fullName);
    router.push("/chat");
  };

  const handleServe = (adapter: LocalModel) => {
    setDefaultModel(adapter.fullName);
    router.push("/serve");
  };

  const handleReload = async (adapter: LocalModel) => {
    setReloading(adapter.id);
    try {
      await reloadAdapter(adapter.cliArgument, false);
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to reload adapter");
    } finally {
      setReloading(null);
    }
  };

  const handleDelete = async (adapter: LocalModel) => {
    if (!window.confirm(`Delete adapter ${adapter.displayName}?`)) return;
    try {
      await deleteMutation.mutateAsync(adapter.id);
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to delete adapter");
    }
  };

  const compareAdapterA = adapters.find((a) => a.id === compareA);
  const compareAdapterB = adapters.find((a) => a.id === compareB);

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Layers className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Adapters</h1>
            <p className="text-sm text-muted-foreground">Manage and compare LoRA adapters</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusPill tone="purple" dot={false}>
            {adapters.length} adapters
          </StatusPill>
          <Dialog open={compareOpen} onOpenChange={setCompareOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" disabled={adapters.length < 2}>
                <ArrowLeftRight className="mr-2 h-4 w-4" />
                Compare
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Compare Adapters</DialogTitle>
              </DialogHeader>
              <div className="grid gap-4 md:grid-cols-2">
                <Select value={compareA} onValueChange={setCompareA}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select adapter A" />
                  </SelectTrigger>
                  <SelectContent>
                    {adapters.map((adapter) => (
                      <SelectItem key={adapter.id} value={adapter.id}>
                        {adapter.displayName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={compareB} onValueChange={setCompareB}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select adapter B" />
                  </SelectTrigger>
                  <SelectContent>
                    {adapters.map((adapter) => (
                      <SelectItem key={adapter.id} value={adapter.id}>
                        {adapter.displayName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                {[compareAdapterA, compareAdapterB].map((adapter, index) => {
                  if (!adapter) {
                    return (
                      <Card key={index}>
                        <CardHeader>
                          <CardTitle>Select an adapter</CardTitle>
                          <CardDescription>Choose an entry to compare</CardDescription>
                        </CardHeader>
                      </Card>
                    );
                  }
                  const summary = adapterSummary(adapter);
                  return (
                    <Card key={adapter.id}>
                      <CardHeader>
                        <CardTitle className="text-base">{adapter.displayName}</CardTitle>
                        <CardDescription>{adapter.baseModel || "Unknown base"}</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm text-muted-foreground">
                        <div className="flex justify-between">
                          <span>Training Mode</span>
                          <span>{summary.trainingMode}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Iterations</span>
                          <span>{summary.iterations ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Best Score</span>
                          <span>{summary.bestScore?.toFixed(3) ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>LoRA Rank</span>
                          <span>{summary.loraRank ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>LoRA Alpha</span>
                          <span>{summary.loraAlpha ?? "—"}</span>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Filter</CardTitle>
              <CardDescription>Filter adapters by base model</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 md:flex-row md:items-center">
              <Select value={selectedBase} onValueChange={setSelectedBase}>
                <SelectTrigger className="md:w-[320px]">
                  <SelectValue placeholder="Select base model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All base models</SelectItem>
                  {baseModels.map((base) => (
                    <SelectItem key={base} value={base}>
                      {base}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </CardContent>
          </Card>

          {isLoading && (
            <div className="flex h-40 items-center justify-center text-muted-foreground">
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Loading adapters…
            </div>
          )}

          {!isLoading && filtered.length === 0 && (
            <Card>
              <CardHeader>
                <CardTitle>No adapters found</CardTitle>
                <CardDescription>Train a model with LoRA to populate this list.</CardDescription>
              </CardHeader>
            </Card>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {filtered.map((adapter) => {
              const summary = adapterSummary(adapter);
              return (
                <Card key={adapter.id} className="flex h-full flex-col">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <IconBadge tone="purple">
                        <Wand2 className="h-4 w-4" />
                      </IconBadge>
                      {adapter.displayName}
                    </CardTitle>
                    <CardDescription>{adapter.baseModel || "Unknown base model"}</CardDescription>
                  </CardHeader>
                  <CardContent className="mt-auto space-y-2 text-sm text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Training</span>
                      <span>{summary.trainingMode}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Iterations</span>
                      <span>{summary.iterations ?? "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Score</span>
                      <span>{summary.bestScore?.toFixed(3) ?? "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Rank / Alpha</span>
                      <span>
                        {summary.loraRank ?? "—"} / {summary.loraAlpha ?? "—"}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2 pt-2">
                      <Button variant="outline" size="sm" onClick={() => handleChat(adapter)}>
                        <MessageSquare className="mr-2 h-4 w-4" />
                        Chat
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => handleServe(adapter)}>
                        <Server className="mr-2 h-4 w-4" />
                        Serve
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => handleReload(adapter)}>
                        <RefreshCw className={reloading === adapter.id ? "mr-2 h-4 w-4 animate-spin" : "mr-2 h-4 w-4"} />
                        Reload
                      </Button>
                      <Button variant="ghost" size="sm" className="text-destructive" onClick={() => handleDelete(adapter)}>
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
