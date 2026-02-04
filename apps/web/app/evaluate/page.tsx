"use client";

import { useEffect, useMemo, useState } from "react";
import { Gauge, RefreshCw, ClipboardList } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { StatusPill } from "@/components/ui/status-pill";
import { CommandCard } from "@/components/ui/command-card";
import { useModels } from "@/hooks/useModels";
import { useSettingsStore } from "@/stores/settings";

interface EvalSummaryRow {
  task_id: string;
  k: number;
  ["pass@k"]: number;
  latency_s: number;
}

interface EvalResults {
  model: string;
  suite: string;
  ts: string;
  summary: EvalSummaryRow[];
}

export default function EvaluatePage() {
  const { data: models } = useModels();
  const { apiUrl } = useSettingsStore();
  const [modelId, setModelId] = useState("");
  const [adapterPath, setAdapterPath] = useState("");
  const [compareAdapterPath, setCompareAdapterPath] = useState("");
  const [suitePath, setSuitePath] = useState("eval/suites/coding.yaml");
  const [results, setResults] = useState<EvalResults | null>(null);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [loadingResults, setLoadingResults] = useState(false);

  const baseModels = useMemo(() => (models || []).filter((m) => !m.isAdapter), [models]);
  const adapterModels = useMemo(() => (models || []).filter((m) => m.isAdapter), [models]);

  useEffect(() => {
    if (!modelId && baseModels.length > 0) {
      setModelId(baseModels[0].cliArgument);
    }
  }, [modelId, baseModels]);

  const modelSpec = adapterPath || modelId;
  const evalCommand = `mlxsmith eval --suite ${suitePath} --model ${modelSpec || "<model-path>"}`;
  const compareCommand =
    compareAdapterPath && modelId
      ? `mlxsmith eval --suite ${suitePath} --model ${compareAdapterPath}`
      : null;

  const loadResults = async () => {
    setLoadingResults(true);
    setResultsError(null);
    try {
      const res = await fetch(`${apiUrl.replace(/\/+$/, "")}/eval/last/results.json`, {
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(`Results not found (HTTP ${res.status})`);
      }
      const data = (await res.json()) as EvalResults;
      setResults(data);
    } catch (err) {
      setResults(null);
      setResultsError(err instanceof Error ? err.message : "Unable to load results");
    } finally {
      setLoadingResults(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Gauge className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Evaluate</h1>
            <p className="text-sm text-muted-foreground">Run evaluation suites and compare adapters</p>
          </div>
        </div>
        <StatusPill tone="blue" dot={false}>
          CLI-driven
        </StatusPill>
      </header>

      <ScrollArea className="flex-1 p-6">
        <Tabs defaultValue="config" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="config">Configure</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="config" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Selection</CardTitle>
                <CardDescription>Choose a base model and optional adapter</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-xs text-muted-foreground">Base Model</label>
                  <Select value={modelId} onValueChange={setModelId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {baseModels.map((model) => (
                        <SelectItem key={model.id} value={model.cliArgument}>
                          {model.displayName}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-muted-foreground">Adapter (optional)</label>
                  <Select value={adapterPath} onValueChange={setAdapterPath}>
                    <SelectTrigger>
                      <SelectValue placeholder="None" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">None</SelectItem>
                      {adapterModels.map((model) => (
                        <SelectItem key={model.id} value={model.cliArgument}>
                          {model.displayName}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Evaluation Suite</CardTitle>
                <CardDescription>Point to a YAML suite under eval/suites</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Input value={suitePath} onChange={(e) => setSuitePath(e.target.value)} />
                <p className="text-xs text-muted-foreground">
                  Example: eval/suites/coding.yaml or eval/suites/rlm_bench.yaml
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Compare Adapter</CardTitle>
                <CardDescription>Optional second run for a comparison baseline</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Select value={compareAdapterPath} onValueChange={setCompareAdapterPath}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select adapter to compare" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">None</SelectItem>
                    {adapterModels.map((model) => (
                      <SelectItem key={model.id} value={model.cliArgument}>
                        {model.displayName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            <CommandCard
              title="Run Evaluation"
              description="Copy the CLI command and run it in your project root."
              command={evalCommand}
              hint="The command writes results to eval/last/results.json."
            />

            {compareCommand && (
              <CommandCard
                title="Compare Adapter"
                description="Run this command after the base evaluation."
                command={compareCommand}
              />
            )}
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <div>
                  <CardTitle>Latest Results</CardTitle>
                  <CardDescription>Fetched from {apiUrl}/eval/last/results.json</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={loadResults} disabled={loadingResults}>
                  <RefreshCw className={loadingResults ? "mr-2 h-4 w-4 animate-spin" : "mr-2 h-4 w-4"} />
                  Load
                </Button>
              </CardHeader>
              <CardContent className="space-y-4">
                {resultsError && (
                  <div className="text-sm text-muted-foreground">{resultsError}</div>
                )}
                {!results && !resultsError && (
                  <div className="text-sm text-muted-foreground">
                    Run an evaluation and click Load to view results.
                  </div>
                )}
                {results && (
                  <div className="space-y-4">
                    <div className="text-sm text-muted-foreground">
                      Model: <span className="text-foreground">{results.model}</span>
                      <br />
                      Suite: <span className="text-foreground">{results.suite}</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border/60 text-left">
                            <th className="py-2">Task</th>
                            <th className="py-2">k</th>
                            <th className="py-2">pass@k</th>
                            <th className="py-2">Latency (s)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.summary.map((row) => (
                            <tr key={row.task_id} className="border-b border-border/60 last:border-0">
                              <td className="py-2">{row.task_id}</td>
                              <td className="py-2">{row.k}</td>
                              <td className="py-2">{row["pass@k"].toFixed(3)}</td>
                              <td className="py-2">{row.latency_s.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Evaluation History</CardTitle>
                <CardDescription>MLXSmith stores evaluation runs in the project workspace.</CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Use the CLI to inspect runs under eval/ and runs/. The web app can display the latest
                results when the API server serves eval/last/results.json.
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </ScrollArea>
    </div>
  );
}
