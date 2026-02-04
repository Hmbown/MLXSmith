"use client";

import { useState } from "react";
import { Cpu, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ModelCard } from "@/components/models/model-card";
import { PullModelDialog } from "@/components/models/pull-model-dialog";
import { useDeleteModel, useModels } from "@/hooks/useModels";
import { useRouter } from "next/navigation";
import { useSettingsStore } from "@/stores/settings";
import { StatusPill } from "@/components/ui/status-pill";
import { formatBytes } from "@/lib/utils";

export default function DownloadedPage() {
  const router = useRouter();
  const { data: models, isLoading, refetch } = useModels();
  const deleteMutation = useDeleteModel();
  const { setDefaultModel } = useSettingsStore();
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const totalSizeBytes = models?.reduce((acc, model) => acc + (model.size_bytes || 0), 0) || 0;

  const handleChat = (model: { fullName: string }) => {
    setDefaultModel(model.fullName);
    router.push("/chat");
  };

  const handleTrain = (model: { fullName: string }) => {
    setDefaultModel(model.fullName);
    router.push("/train");
  };

  const handleServe = (model: { fullName: string }) => {
    setDefaultModel(model.fullName);
    router.push("/serve");
  };

  const handleDelete = async (modelId: string) => {
    if (!window.confirm("Delete this model from the cache?")) return;
    setPendingDelete(modelId);
    try {
      await deleteMutation.mutateAsync(modelId);
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to delete model");
    }
    setPendingDelete(null);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Cpu className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Downloaded</h1>
            <p className="text-sm text-muted-foreground">Manage cached MLX and HF models</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusPill tone="blue" dot={false}>
            {models?.length || 0} models • {formatBytes(totalSizeBytes)}
          </StatusPill>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <PullModelDialog triggerVariant="glow" />
        </div>
      </header>

      <ScrollArea className="flex-1 p-6">
        {isLoading ? (
          <div className="flex h-64 items-center justify-center">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : !models || models.length === 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>No models found</CardTitle>
              <CardDescription>
                Pull your first model from HuggingFace to get started.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PullModelDialog triggerVariant="glow" />
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                onChat={handleChat}
                onTrain={handleTrain}
                onServe={handleServe}
                onDelete={() => handleDelete(model.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>

      {pendingDelete && (
        <div className="pointer-events-none fixed bottom-6 right-6 flex items-center gap-2 rounded-full border border-border/60 bg-card/80 px-4 py-2 text-xs text-muted-foreground shadow-soft backdrop-blur">
          <Trash2 className="h-3 w-3" />
          Deleting model…
        </div>
      )}
    </div>
  );
}
