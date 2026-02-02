"use client";

import { useQuery } from "@tanstack/react-query";
import { getRLMState, getRLMHistory } from "@/lib/api";

const RLM_STATE_KEY = "rlm-state";
const RLM_HISTORY_KEY = "rlm-history";

export function useRLMState() {
  return useQuery({
    queryKey: [RLM_STATE_KEY],
    queryFn: getRLMState,
    refetchInterval: 5000, // Refetch every 5 seconds
  });
}

export function useRLMHistory() {
  return useQuery({
    queryKey: [RLM_HISTORY_KEY],
    queryFn: getRLMHistory,
    refetchInterval: 5000,
  });
}
