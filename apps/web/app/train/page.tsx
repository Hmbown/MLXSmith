"use client";

import { Dumbbell, Activity } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RLMCharts } from "@/components/train/rlm-charts";
import { TrainingConfig } from "@/components/train/training-config";
import { useRLMState, useRLMHistory } from "@/hooks/useRLM";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDate } from "@/lib/utils";

export default function TrainPage() {
  const { data: rlmState, isLoading: stateLoading } = useRLMState();
  const { data: rlmHistory, isLoading: historyLoading } = useRLMHistory();

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "running":
        return "bg-green-500";
      case "paused":
        return "bg-yellow-500";
      case "error":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-16 items-center justify-between border-b px-6">
        <div className="flex items-center gap-4">
          <Dumbbell className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Train</h1>
            <p className="text-sm text-muted-foreground">
              RLM training and fine-tuning
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 rounded-full border px-3 py-1">
            <div className={`h-2 w-2 rounded-full ${getStatusColor(rlmState?.status)} animate-pulse`} />
            <span className="text-sm capitalize">{rlmState?.status || "idle"}</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <ScrollArea className="flex-1 p-6">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Config */}
          <div className="space-y-6">
            <TrainingConfig />
          </div>

          {/* Right Column - Charts & Status */}
          <div className="space-y-6 lg:col-span-2">
            {/* Status Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Training Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                {stateLoading ? (
                  <div className="text-muted-foreground">Loading...</div>
                ) : !rlmState || Object.keys(rlmState).length === 0 ? (
                  <div className="text-muted-foreground">No active training session</div>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">Status</p>
                      <Badge variant={rlmState.status === "running" ? "default" : "secondary"}>
                        {rlmState.status}
                      </Badge>
                    </div>
                    {rlmState.currentStep !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Step</p>
                        <p className="font-medium">
                          {rlmState.currentStep} / {rlmState.totalSteps || "?"}
                        </p>
                      </div>
                    )}
                    {rlmState.currentLoss !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Current Loss</p>
                        <p className="font-medium">{rlmState.currentLoss.toFixed(4)}</p>
                      </div>
                    )}
                    {rlmState.bestScore !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Best Score</p>
                        <p className="font-medium">{rlmState.bestScore.toFixed(4)}</p>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Charts */}
            <RLMCharts history={rlmHistory || []} />

            {/* History Table */}
            <Card>
              <CardHeader>
                <CardTitle>Recent History</CardTitle>
                <CardDescription>Last 10 training iterations</CardDescription>
              </CardHeader>
              <CardContent>
                {!rlmHistory || rlmHistory.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No history available
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="py-2 text-left font-medium">Step</th>
                          <th className="py-2 text-left font-medium">Loss</th>
                          <th className="py-2 text-left font-medium">Reward</th>
                          <th className="py-2 text-left font-medium">Score</th>
                          <th className="py-2 text-left font-medium">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rlmHistory.slice(-10).reverse().map((entry) => (
                          <tr key={entry.step} className="border-b last:border-0">
                            <td className="py-2">{entry.step}</td>
                            <td className="py-2">{entry.loss?.toFixed(4) || "-"}</td>
                            <td className="py-2">{entry.reward?.toFixed(4) || "-"}</td>
                            <td className="py-2">{entry.adapter_score?.toFixed(4) || "-"}</td>
                            <td className="py-2">
                              <Badge variant={entry.passed ? "default" : "secondary"}>
                                {entry.passed ? "Passed" : "Failed"}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
