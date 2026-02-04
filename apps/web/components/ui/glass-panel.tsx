import type React from "react";
import { cn } from "@/lib/utils";

interface GlassPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  inset?: boolean;
}

export function GlassPanel({ className, inset, ...props }: GlassPanelProps) {
  return (
    <div
      className={cn(
        "glass-panel shadow-soft",
        inset && "bg-card/40",
        className
      )}
      {...props}
    />
  );
}
