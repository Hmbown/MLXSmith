"use client";

import { useState } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useModels } from "@/hooks/useModels";
import type { TrainingConfig as TrainingConfigType } from "@/types";

interface TrainingConfigProps {
  onStart?: (config: TrainingConfigType) => void;
  onStop?: () => void;
  onReset?: () => void;
  isTraining?: boolean;
}

export function TrainingConfig({ onStart, onStop, onReset, isTraining }: TrainingConfigProps) {
  const { data: models } = useModels();
  const [config, setConfig] = useState<Partial<TrainingConfigType>>({
    epochs: 3,
    batchSize: 4,
    learningRate: 0.0001,
    loraR: 8,
    loraAlpha: 16,
    loraDropout: 0.1,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Training Configuration</CardTitle>
        <CardDescription>
          Configure your training run parameters
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Model Selection */}
        <div className="space-y-2">
          <Label>Base Model</Label>
          <Select
            value={config.model}
            onValueChange={(v) => setConfig((c) => ({ ...c, model: v }))}
            disabled={isTraining}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {models?.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Dataset Path */}
        <div className="space-y-2">
          <Label>Dataset Path</Label>
          <Input
            placeholder="path/to/dataset.jsonl"
            value={config.dataset || ""}
            onChange={(e) => setConfig((c) => ({ ...c, dataset: e.target.value }))}
            disabled={isTraining}
          />
        </div>

        {/* Output Path */}
        <div className="space-y-2">
          <Label>Output Directory</Label>
          <Input
            placeholder="adapters/my-adapter"
            value={config.output || ""}
            onChange={(e) => setConfig((c) => ({ ...c, output: e.target.value }))}
            disabled={isTraining}
          />
        </div>

        {/* Epochs */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label>Epochs</Label>
            <span className="text-sm text-muted-foreground">{config.epochs}</span>
          </div>
          <Slider
            value={[config.epochs || 3]}
            onValueChange={([v]) => setConfig((c) => ({ ...c, epochs: v }))}
            min={1}
            max={10}
            step={1}
            disabled={isTraining}
          />
        </div>

        {/* Batch Size */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label>Batch Size</Label>
            <span className="text-sm text-muted-foreground">{config.batchSize}</span>
          </div>
          <Slider
            value={[config.batchSize || 4]}
            onValueChange={([v]) => setConfig((c) => ({ ...c, batchSize: v }))}
            min={1}
            max={32}
            step={1}
            disabled={isTraining}
          />
        </div>

        {/* Learning Rate */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <Label>Learning Rate</Label>
            <span className="text-sm text-muted-foreground">{config.learningRate}</span>
          </div>
          <Slider
            value={[config.learningRate ? Math.log10(config.learningRate) : -4]}
            onValueChange={([v]) => setConfig((c) => ({ ...c, learningRate: Math.pow(10, v) }))}
            min={-6}
            max={-2}
            step={0.5}
            disabled={isTraining}
          />
        </div>

        {/* LoRA Settings */}
        <div className="space-y-4 rounded-lg border p-4">
          <h4 className="font-medium">LoRA Configuration</h4>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Rank (r)</Label>
              <span className="text-sm text-muted-foreground">{config.loraR}</span>
            </div>
            <Slider
              value={[config.loraR || 8]}
              onValueChange={([v]) => setConfig((c) => ({ ...c, loraR: v }))}
              min={4}
              max={64}
              step={4}
              disabled={isTraining}
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Alpha</Label>
              <span className="text-sm text-muted-foreground">{config.loraAlpha}</span>
            </div>
            <Slider
              value={[config.loraAlpha || 16]}
              onValueChange={([v]) => setConfig((c) => ({ ...c, loraAlpha: v }))}
              min={8}
              max={128}
              step={8}
              disabled={isTraining}
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Dropout</Label>
              <span className="text-sm text-muted-foreground">{config.loraDropout}</span>
            </div>
            <Slider
              value={[config.loraDropout || 0.1]}
              onValueChange={([v]) => setConfig((c) => ({ ...c, loraDropout: v }))}
              min={0}
              max={0.5}
              step={0.05}
              disabled={isTraining}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-4">
          {isTraining ? (
            <Button variant="destructive" className="flex-1" onClick={onStop}>
              <Pause className="mr-2 h-4 w-4" />
              Stop Training
            </Button>
          ) : (
            <Button
              className="flex-1"
              onClick={() => onStart?.(config as TrainingConfigType)}
              disabled={!config.model || !config.dataset}
            >
              <Play className="mr-2 h-4 w-4" />
              Start Training
            </Button>
          )}
          <Button variant="outline" size="icon" onClick={onReset} disabled={isTraining}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
