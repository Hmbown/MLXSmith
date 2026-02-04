import type React from "react";
import { cn } from "@/lib/utils";

interface StatusPillProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: "blue" | "green" | "yellow" | "red" | "neutral" | "purple";
  dot?: boolean;
}

const toneClasses: Record<NonNullable<StatusPillProps["tone"]>, string> = {
  blue: "text-blue-600 dark:text-blue-300 border-blue-200/60 bg-blue-100/50",
  green: "text-emerald-600 dark:text-emerald-300 border-emerald-200/60 bg-emerald-100/50",
  yellow: "text-amber-600 dark:text-amber-300 border-amber-200/60 bg-amber-100/50",
  red: "text-rose-600 dark:text-rose-300 border-rose-200/60 bg-rose-100/50",
  purple: "text-purple-600 dark:text-purple-300 border-purple-200/60 bg-purple-100/50",
  neutral: "text-muted-foreground border-border/70 bg-muted/60",
};

export function StatusPill({ className, tone = "neutral", dot = true, children, ...props }: StatusPillProps) {
  return (
    <span className={cn("status-pill border", toneClasses[tone], className)} {...props}>
      {dot && <span className="h-2 w-2 rounded-full bg-current" />}
      <span>{children}</span>
    </span>
  );
}
