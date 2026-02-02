"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { saveHFToken, getHFToken, saveProjectPath, getProjectPath } from "@/lib/api";
import { useSettingsStore } from "@/stores/settings";

const HF_TOKEN_KEY = "hf-token";
const PROJECT_PATH_KEY = "project-path";

export function useHFToken() {
  const queryClient = useQueryClient();
  const { hfToken, setHfToken: setStoreToken } = useSettingsStore();

  const { data } = useQuery({
    queryKey: [HF_TOKEN_KEY],
    queryFn: getHFToken,
    enabled: false, // Don't auto-fetch, use local storage
  });

  const mutation = useMutation({
    mutationFn: saveHFToken,
    onSuccess: (_, variables) => {
      setStoreToken(variables);
      queryClient.invalidateQueries({ queryKey: [HF_TOKEN_KEY] });
    },
  });

  return {
    token: hfToken,
    setToken: mutation.mutate,
    isLoading: mutation.isPending,
  };
}

export function useProjectPath() {
  const queryClient = useQueryClient();
  const { projectPath, setProjectPath: setStorePath } = useSettingsStore();

  const { data } = useQuery({
    queryKey: [PROJECT_PATH_KEY],
    queryFn: getProjectPath,
    enabled: false,
  });

  const mutation = useMutation({
    mutationFn: saveProjectPath,
    onSuccess: (_, variables) => {
      setStorePath(variables);
      queryClient.invalidateQueries({ queryKey: [PROJECT_PATH_KEY] });
    },
  });

  return {
    path: projectPath,
    setPath: mutation.mutate,
    isLoading: mutation.isPending,
  };
}
