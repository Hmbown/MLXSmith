"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listModels, pullModel } from "@/lib/api";

const MODELS_KEY = "models";

export function useModels() {
  return useQuery({
    queryKey: [MODELS_KEY],
    queryFn: listModels,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

export function usePullModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: pullModel,
    onSuccess: () => {
      // Invalidate and refetch models list
      queryClient.invalidateQueries({ queryKey: [MODELS_KEY] });
    },
  });
}
