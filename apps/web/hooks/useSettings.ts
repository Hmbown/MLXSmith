"use client";

import { useMutation } from "@tanstack/react-query";
import { saveHFToken } from "@/lib/api";
import { useSettingsStore } from "@/stores/settings";

export function useHFToken() {
  const { setHfToken: setStoreToken } = useSettingsStore();

  const mutation = useMutation({
    mutationFn: (token: string) => saveHFToken(token, true, true),
    onSuccess: (_, variables) => {
      setStoreToken(variables);
    },
  });

  return {
    setToken: mutation.mutateAsync,
    isLoading: mutation.isPending,
  };
}
