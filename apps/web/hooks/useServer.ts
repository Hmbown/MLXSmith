"use client";

import { useQuery } from "@tanstack/react-query";
import { getServerStatus } from "@/lib/api";
import { useServerStore } from "@/stores/server";

const SERVER_STATUS_KEY = "server-status";

export function useServerStatus() {
  const { setRunning } = useServerStore();

  return useQuery({
    queryKey: [SERVER_STATUS_KEY],
    queryFn: async () => {
      const status = await getServerStatus();
      setRunning(status.running);
      return status;
    },
    refetchInterval: 5000,
  });
}
