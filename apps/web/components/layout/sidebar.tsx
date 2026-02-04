"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brain, Command } from "lucide-react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatusPill } from "@/components/ui/status-pill";
import { navigationSections, type NavItem } from "@/components/layout/navigation";


function NavLink({ item, isActive }: { item: NavItem; isActive: boolean }) {
  return (
    <Link
      href={item.href}
      className={cn(
        "flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-colors",
        isActive
          ? "bg-primary/90 text-primary-foreground shadow-soft"
          : "text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground"
      )}
    >
      {item.icon}
      {item.title}
    </Link>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col border-r border-border/60 bg-card/40 backdrop-blur">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border/60 px-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-soft">
          <Brain className="h-5 w-5" />
        </div>
        <div>
          <span className="text-lg font-semibold">MLXSmith</span>
          <div className="text-xs text-muted-foreground">Studio Console</div>
        </div>
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 px-3 py-4">
        <div className="space-y-4">
          {navigationSections.map((section) => (
            <div key={section.label} className="space-y-1">
              <div className="px-3 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {section.label}
              </div>
              {section.items.map((item) => (
                <NavLink
                  key={item.href}
                  item={item}
                  isActive={pathname === item.href || pathname.startsWith(`${item.href}/`)}
                />
              ))}
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="border-t border-border/60 p-4">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-2">
            <Command className="h-3 w-3" />
            <span>MLXSmith v0.1.0</span>
          </span>
          <StatusPill tone="blue" dot={false}>
            Local API
          </StatusPill>
        </div>
      </div>
    </div>
  );
}
