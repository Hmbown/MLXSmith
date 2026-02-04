"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { navigationSections } from "@/components/layout/navigation";
import { cn } from "@/lib/utils";

export function MobileNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <div className="flex items-center justify-between border-b border-border/60 bg-card/40 px-4 py-3 backdrop-blur md:hidden">
      <div className="text-sm font-semibold">MLXSmith</div>
      <Button variant="ghost" size="icon" onClick={() => setOpen(true)}>
        <Menu className="h-5 w-5" />
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-sm p-0">
          <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
            <span className="text-sm font-semibold">Navigate</span>
            <Button variant="ghost" size="icon" onClick={() => setOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="max-h-[70vh] overflow-y-auto p-4">
            {navigationSections.map((section) => (
              <div key={section.label} className="mb-4 space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {section.label}
                </div>
                <div className="space-y-1">
                  {section.items.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setOpen(false)}
                      className={cn(
                        "flex items-center gap-3 rounded-xl px-3 py-2 text-sm",
                        pathname === item.href || pathname.startsWith(`${item.href}/`)
                          ? "bg-primary/90 text-primary-foreground"
                          : "text-muted-foreground hover:bg-accent/40 hover:text-accent-foreground"
                      )}
                    >
                      {item.icon}
                      {item.title}
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
