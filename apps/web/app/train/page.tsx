"use client";

import { useEffect, useMemo, useState } from "react";
import { Dumbbell, Database, Copy } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusPill } from "@/components/ui/status-pill";
import { CommandCard } from "@/components/ui/command-card";
import { useModels } from "@/hooks/useModels";
import { useRLMState, useRLMHistory } from "@/hooks/useRLM";
import { RLMCharts } from "@/components/train/rlm-charts";

type TrainingMode = "sft" | "pref" | "rft" | "rlm";

const preferenceLossTypes = ["dpo", "orpo", "ipo", "cpo", "simpo", "hinge", "tdpo"];
const rftLossTypes = ["grpo", "dr_grpo", "dapo"];

export default function TrainPage() {
  const { data: models } = useModels();
  const { data: rlmState } = useRLMState();
  const { data: rlmHistory } = useRLMHistory();

  const [mode, setMode] = useState<TrainingMode>("sft");
  const [modelId, setModelId] = useState("");
  const [datasetPath, setDatasetPath] = useState("data/sft");
  const [outputPath, setOutputPath] = useState("runs/sft_0001");
  const [learningRate, setLearningRate] = useState(2e-4);
  const [batchSize, setBatchSize] = useState(1);
  const [gradientAccum, setGradientAccum] = useState(8);
  const [seed, setSeed] = useState(1337);
  const [iterations, setIterations] = useState(1000);
  const [loraRank, setLoraRank] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [prefLossType, setPrefLossType] = useState("dpo");
  const [rftEnv, setRftEnv] = useState("envs/coding.yaml");
  const [rftVerifier, setRftVerifier] = useState("verifiers/regex.py");
  const [rftRollouts, setRftRollouts] = useState(8);
  const [rftLossType, setRftLossType] = useState("grpo");
  const [rftEpsilonLow, setRftEpsilonLow] = useState(0.2);
  const [rftEpsilonHigh, setRftEpsilonHigh] = useState(0.2);
  const [rftTokenLoss, setRftTokenLoss] = useState(false);
  const [rlmIterations, setRlmIterations] = useState(50);
  const [rlmResume, setRlmResume] = useState(false);
  const [rlmOrchestrated, setRlmOrchestrated] = useState(false);

  const baseModels = useMemo(() => models || [], [models]);

  useEffect(() => {
    if (!modelId && baseModels.length > 0) {
      setModelId(baseModels[0].cliArgument);
    }
  }, [modelId, baseModels]);

  const trainingCommand = useMemo(() => {
    if (!modelId) {
      return "mlxsmith <command> --model <model-path>";
    }
    switch (mode) {
      case "sft":
        return `mlxsmith sft --model ${modelId} --data ${datasetPath} --lr ${learningRate} --iters ${iterations} --batch-size ${batchSize}`;
      case "pref":
        return `mlxsmith pref --model ${modelId} --data ${datasetPath} --loss-type ${prefLossType}`;
      case "rft":
        return `mlxsmith rft --model ${modelId} --env ${rftEnv} --verifier ${rftVerifier} --rollouts ${rftRollouts} --loss-type ${rftLossType} --epsilon-low ${rftEpsilonLow} --epsilon-high ${rftEpsilonHigh}${rftTokenLoss ? " --token-level-loss" : ""}`;
      case "rlm":
        return `mlxsmith rlm --model ${modelId} --iterations ${rlmIterations}${rlmResume ? " --resume" : ""}${rlmOrchestrated ? " --orchestrated" : ""}`;
      default:
        return "mlxsmith <command>";
    }
  }, [
    mode,
    modelId,
    datasetPath,
    learningRate,
    iterations,
    batchSize,
    prefLossType,
    rftEnv,
    rftVerifier,
    rftRollouts,
    rftLossType,
    rftEpsilonLow,
    rftEpsilonHigh,
    rftTokenLoss,
    rlmIterations,
    rlmResume,
    rlmOrchestrated,
  ]);

  const envOverrides = [
    `MLXSMITH__TRAIN__GRAD_ACCUM=${gradientAccum}`,
    `MLXSMITH__TRAIN__SEED=${seed}`,
    `MLXSMITH__LORA__R=${loraRank}`,
    `MLXSMITH__LORA__ALPHA=${loraAlpha}`,
    `MLXSMITH__LORA__DROPOUT=${loraDropout}`,
  ].join(" ");

  const fullCommand = `${envOverrides} ${trainingCommand}`;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Dumbbell className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Train</h1>
            <p className="text-sm text-muted-foreground">CLI-aligned training workflows</p>
          </div>
        </div>
        <StatusPill tone={rlmState?.status === "running" ? "green" : "neutral"}>
          {rlmState?.status || "idle"}
        </StatusPill>
      </header>

      <ScrollArea className="flex-1 p-6">
        <Tabs defaultValue="config" className="space-y-6">
          <TabsList className="grid w-full max-w-lg grid-cols-3">
            <TabsTrigger value="config">Configure</TabsTrigger>
            <TabsTrigger value="monitor">Monitor</TabsTrigger>
            <TabsTrigger value="datasets">Datasets</TabsTrigger>
          </TabsList>

          <TabsContent value="config" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Training Mode</CardTitle>
                <CardDescription>Choose the workflow to run</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2">
                <Select value={mode} onValueChange={(v) => setMode(v as TrainingMode)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sft">SFT (LoRA)</SelectItem>
                    <SelectItem value="pref">Preference</SelectItem>
                    <SelectItem value="rft">RFT</SelectItem>
                    <SelectItem value="rlm">RLM Loop</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={modelId} onValueChange={setModelId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {baseModels.map((model) => (
                      <SelectItem key={model.id} value={model.cliArgument}>
                        {model.displayName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {(mode === "sft" || mode === "pref") && (
              <Card>
                <CardHeader>
                  <CardTitle>Dataset</CardTitle>
                  <CardDescription>Dataset folder containing train.jsonl</CardDescription>
                </CardHeader>
                <CardContent>
                  <Input value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} />
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle>Output</CardTitle>
                <CardDescription>Optional run name and destination (manual move after CLI run)</CardDescription>
              </CardHeader>
              <CardContent>
                <Input value={outputPath} onChange={(e) => setOutputPath(e.target.value)} />
              </CardContent>
            </Card>

            {mode === "sft" && (
              <Card>
                <CardHeader>
                  <CardTitle>Training Parameters</CardTitle>
                  <CardDescription>Core SFT hyperparameters</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Learning Rate</Label>
                    <Input
                      type="number"
                      value={learningRate}
                      step={0.00001}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Iterations</Label>
                    <Input
                      type="number"
                      value={iterations}
                      onChange={(e) => setIterations(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Batch Size</Label>
                    <Input
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Gradient Accumulation</Label>
                    <Input
                      type="number"
                      value={gradientAccum}
                      onChange={(e) => setGradientAccum(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Seed</Label>
                    <Input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {mode === "sft" && (
              <Card>
                <CardHeader>
                  <CardTitle>LoRA Configuration</CardTitle>
                  <CardDescription>Adapters and rank settings</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label>Rank</Label>
                    <Input
                      type="number"
                      value={loraRank}
                      onChange={(e) => setLoraRank(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Alpha</Label>
                    <Input
                      type="number"
                      value={loraAlpha}
                      onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Dropout</Label>
                    <Input
                      type="number"
                      step={0.01}
                      value={loraDropout}
                      onChange={(e) => setLoraDropout(Number(e.target.value))}
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {mode === "pref" && (
              <Card>
                <CardHeader>
                  <CardTitle>Preference Settings</CardTitle>
                  <CardDescription>DPO / ORPO loss configuration</CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={prefLossType} onValueChange={setPrefLossType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select loss type" />
                    </SelectTrigger>
                    <SelectContent>
                      {preferenceLossTypes.map((loss) => (
                        <SelectItem key={loss} value={loss}>
                          {loss}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>
            )}

            {mode === "rft" && (
              <Card>
                <CardHeader>
                  <CardTitle>RFT Settings</CardTitle>
                  <CardDescription>Environment + verifier paths</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Environment YAML</Label>
                    <Input value={rftEnv} onChange={(e) => setRftEnv(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label>Verifier Script</Label>
                    <Input value={rftVerifier} onChange={(e) => setRftVerifier(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label>Rollouts</Label>
                    <Input
                      type="number"
                      value={rftRollouts}
                      onChange={(e) => setRftRollouts(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Loss Type</Label>
                    <Select value={rftLossType} onValueChange={setRftLossType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select loss type" />
                      </SelectTrigger>
                      <SelectContent>
                        {rftLossTypes.map((loss) => (
                          <SelectItem key={loss} value={loss}>
                            {loss}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Epsilon Low</Label>
                    <Input
                      type="number"
                      step={0.01}
                      value={rftEpsilonLow}
                      onChange={(e) => setRftEpsilonLow(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Epsilon High</Label>
                    <Input
                      type="number"
                      step={0.01}
                      value={rftEpsilonHigh}
                      onChange={(e) => setRftEpsilonHigh(Number(e.target.value))}
                    />
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border/60 bg-background/60 px-3 py-2">
                    <span className="text-sm">Token-level loss</span>
                    <Switch checked={rftTokenLoss} onCheckedChange={setRftTokenLoss} />
                  </div>
                </CardContent>
              </Card>
            )}

            {mode === "rlm" && (
              <Card>
                <CardHeader>
                  <CardTitle>RLM Loop</CardTitle>
                  <CardDescription>Recursive training configuration</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Iterations</Label>
                    <Input
                      type="number"
                      value={rlmIterations}
                      onChange={(e) => setRlmIterations(Number(e.target.value))}
                    />
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border/60 bg-background/60 px-3 py-2">
                    <span className="text-sm">Resume</span>
                    <Switch checked={rlmResume} onCheckedChange={setRlmResume} />
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border/60 bg-background/60 px-3 py-2">
                    <span className="text-sm">Orchestrated</span>
                    <Switch checked={rlmOrchestrated} onCheckedChange={setRlmOrchestrated} />
                  </div>
                </CardContent>
              </Card>
            )}

            <CommandCard
              title="Training Command"
              description="Copy the CLI command to run training."
              command={fullCommand}
              hint="Set MLXSMITH__... variables to override config fields, matching the macOS app."
            />
          </TabsContent>

          <TabsContent value="monitor" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>RLM Status</CardTitle>
                <CardDescription>Live state from /internal/rlm/state</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <div className="text-xs text-muted-foreground">Status</div>
                  <div className="text-base font-semibold capitalize">{rlmState?.status || "idle"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Iteration</div>
                  <div className="text-base font-semibold">{rlmState?.iteration ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Total Iterations</div>
                  <div className="text-base font-semibold">{rlmState?.total_iterations ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Loss</div>
                  <div className="text-base font-semibold">{rlmState?.metrics?.loss?.toFixed(4) ?? "—"}</div>
                </div>
              </CardContent>
            </Card>

            <RLMCharts history={rlmHistory || []} />

            <Card>
              <CardHeader>
                <CardTitle>Recent RLM History</CardTitle>
                <CardDescription>Latest iterations from /internal/rlm/history</CardDescription>
              </CardHeader>
              <CardContent>
                {!rlmHistory || rlmHistory.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No history available yet.</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border/60 text-left">
                          <th className="py-2">Iteration</th>
                          <th className="py-2">Adapter Score</th>
                          <th className="py-2">Loss</th>
                          <th className="py-2">Reward</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rlmHistory.slice(-10).reverse().map((entry) => (
                          <tr key={entry.iteration} className="border-b border-border/60 last:border-0">
                            <td className="py-2">{entry.iteration}</td>
                            <td className="py-2">{entry.adapter_score?.toFixed(3) ?? "—"}</td>
                            <td className="py-2">{entry.metrics?.loss?.toFixed(4) ?? "—"}</td>
                            <td className="py-2">{entry.metrics?.reward_mean?.toFixed(4) ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="datasets" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Dataset Commands
                </CardTitle>
                <CardDescription>CLI helpers for data preparation</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 md:grid-cols-2">
                {[
                  "mlxsmith data presets",
                  "mlxsmith data pull --preset alpaca --out-dir data/sft",
                  "mlxsmith data import --in raw.json --format sharegpt --out data/sft/train.jsonl",
                  "mlxsmith data split --in data/sft/train.jsonl --valid 0.05 --test 0.05",
                  "mlxsmith data validate --in data/sft/train.jsonl",
                  "mlxsmith data stats --in data/sft/train.jsonl",
                ].map((cmd) => (
                  <Button
                    key={cmd}
                    variant="outline"
                    className="justify-between text-left"
                    onClick={() => navigator.clipboard.writeText(cmd)}
                  >
                    <span className="truncate">{cmd}</span>
                    <Copy className="h-4 w-4" />
                  </Button>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </ScrollArea>
    </div>
  );
}
