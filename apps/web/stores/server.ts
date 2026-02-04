import { create } from "zustand";

interface ServerState {
  isRunning: boolean;
  setRunning: (running: boolean) => void;
}

export const useServerStore = create<ServerState>((set) => ({
  isRunning: false,
  setRunning: (isRunning) => set({ isRunning }),
}));
