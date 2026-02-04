"use client";

import { useEffect, useMemo, useState } from "react";
import { Compass, Search, RefreshCw, Sparkles, ArrowUpRight } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { StatusPill } from "@/components/ui/status-pill";
import { PullModelDialog } from "@/components/models/pull-model-dialog";
import { cn } from "@/lib/utils";

interface HFModel {
  id: string;
  pipeline_tag?: string;
  tags?: string[];
  downloads?: number;
  likes?: number;
  lastModified?: string;
}

type FilterType = "all" | "mlx" | "text" | "embeddings";

const pageSize = 20;

function isMLX(model: HFModel) {
  const id = model.id.toLowerCase();
  const tags = model.tags?.map((t) => t.toLowerCase()) || [];
  return id.includes("mlx") || tags.some((tag) => tag.includes("mlx"));
}

export default function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterType>("all");
  const [models, setModels] = useState<HFModel[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);

  const fetchModels = async (nextOffset = 0, append = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        limit: String(pageSize),
        offset: String(nextOffset),
      });
      if (query.trim()) {
        params.set("search", query.trim());
      } else {
        params.set("sort", "downloads");
        params.set("direction", "-1");
      }

      const res = await fetch(`https://huggingface.co/api/models?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`HuggingFace error (${res.status})`);
      }
      const data = (await res.json()) as HFModel[];
      setModels((prev) => (append ? [...prev, ...data] : data));
      setOffset(nextOffset);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch models");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const handle = setTimeout(() => {
      fetchModels(0, false);
    }, 350);
    return () => clearTimeout(handle);
  }, [query]);


  const filteredModels = useMemo(() => {
    switch (filter) {
      case "mlx":
        return models.filter(isMLX);
      case "text":
        return models.filter((m) => m.pipeline_tag === "text-generation");
      case "embeddings":
        return models.filter(
          (m) => m.pipeline_tag === "feature-extraction" || m.tags?.includes("embeddings")
        );
      default:
        return models;
    }
  }, [models, filter]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Compass className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Discover</h1>
            <p className="text-sm text-muted-foreground">Browse HuggingFace models</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusPill tone="blue" dot={false}>
            {filteredModels.length} results
          </StatusPill>
          <Button variant="outline" size="sm" onClick={() => fetchModels(0, false)}>
            <RefreshCw className={cn("mr-2 h-4 w-4", isLoading && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Search</CardTitle>
              <CardDescription>Find models by name or tag</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 md:flex-row md:items-center">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search HuggingFace models..."
                  className="pl-9"
                />
              </div>
              <Select value={filter} onValueChange={(v) => setFilter(v as FilterType)}>
                <SelectTrigger className="md:w-[200px]">
                  <SelectValue placeholder="Filter" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All models</SelectItem>
                  <SelectItem value="mlx">MLX-compatible</SelectItem>
                  <SelectItem value="text">Text generation</SelectItem>
                  <SelectItem value="embeddings">Embeddings</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {error && (
            <Card>
              <CardHeader>
                <CardTitle>HuggingFace Error</CardTitle>
                <CardDescription>{error}</CardDescription>
              </CardHeader>
            </Card>
          )}

          {!error && filteredModels.length === 0 && !isLoading && (
            <Card>
              <CardHeader>
                <CardTitle>No models found</CardTitle>
                <CardDescription>Try a different search or filter.</CardDescription>
              </CardHeader>
            </Card>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {filteredModels.map((model) => (
              <Card key={model.id} className="flex h-full flex-col">
                <CardHeader>
                  <CardTitle className="text-base">{model.id}</CardTitle>
                  <CardDescription className="flex flex-wrap gap-2 text-xs">
                    {model.pipeline_tag && (
                      <StatusPill tone="blue" dot={false}>
                        {model.pipeline_tag}
                      </StatusPill>
                    )}
                    {isMLX(model) && (
                      <StatusPill tone="green" dot={false}>
                        MLX
                      </StatusPill>
                    )}
                  </CardDescription>
                </CardHeader>
                <CardContent className="mt-auto space-y-3 text-sm text-muted-foreground">
                  <div className="flex items-center justify-between">
                    <span>Downloads</span>
                    <span>{model.downloads?.toLocaleString() || "—"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Likes</span>
                    <span>{model.likes?.toLocaleString() || "—"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Updated</span>
                    <span>{model.lastModified ? new Date(model.lastModified).toLocaleDateString() : "—"}</span>
                  </div>
                  <div className="flex items-center justify-between pt-2">
                    <PullModelDialog
                      triggerLabel="Pull"
                      triggerVariant="glow"
                      triggerSize="sm"
                      defaultModelId={model.id}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => window.open(`https://huggingface.co/${model.id}`, "_blank")}
                    >
                      View <ArrowUpRight className="ml-1 h-3 w-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="flex justify-center">
            <Button
              variant="outline"
              onClick={() => fetchModels(offset + pageSize, true)}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Loading
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Load More
                </>
              )}
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
