"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getServerStatus, startServer, stopServer } from "@/lib/api";
import { useServerStore } from "@/stores/server";

const SERVER_STATUS_KEY = "server-status";

export function useServerStatus() {
  const { setRunning, setPort } = useServerStore();

  return useQuery({
    queryKey: [SERVER_STATUS_KEY],
    queryFn: async () => {
      const status = await getServerStatus();
      setRunning(status.running);
      setPort(status.port);
      return status;
    },
    refetchInterval: 5000,
  });
}

export function useToggleServer() {
  const queryClient = useQueryClient();
  const { isRunning, port, setRunning, setLoading, setError } = useServerStore();

  const startMutation = useMutation({
    mutationFn: startServer,
    onMutate: () => {
      setLoading(true);
      setError(null);
    },
    onSuccess: () => {
      setRunning(true);
      setLoading(false);
      queryClient.invalidateQueries({ queryKey: [SERVER_STATUS_KEY] });
    },
    onError: (error: Error) => {
      setLoading(false);
      setError(error.message);
    },
  });

  const stopMutation = useMutation({
    mutationFn: stopServer,
    onMutate: () => {
      setLoading(true);
      setError(null);
    },
    onSuccess: () => {
      setRunning(false);
      setLoading(false);
      queryClient.invalidateQueries({ queryKey: [SERVER_STATUS_KEY] });
    },
    onError: (error: Error) => {
      setLoading(false);
      setError(error.message);
    },
  });

  const toggle = () => {
    if (isRunning) {
      stopMutation.mutate();
    } else {
      startMutation.mutate(port);
    }
  };

  return {
    toggle,
    isRunning,
    isLoading: startMutation.isPending || stopMutation.isPending,
    error: useServerStore((s) => s.error),
  };
}
