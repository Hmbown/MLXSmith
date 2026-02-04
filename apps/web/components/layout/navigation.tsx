import type React from "react";
import {
  LayoutDashboard,
  Compass,
  Cpu,
  Layers,
  MessageSquare,
  Dumbbell,
  Gauge,
  Boxes,
  Server,
  Settings,
} from "lucide-react";

export interface NavItem {
  title: string;
  href: string;
  icon: React.ReactNode;
}

export interface NavSection {
  label: string;
  items: NavItem[];
}

export const navigationSections: NavSection[] = [
  {
    label: "Overview",
    items: [
      { title: "Dashboard", href: "/dashboard", icon: <LayoutDashboard className="h-5 w-5" /> },
    ],
  },
  {
    label: "Models",
    items: [
      { title: "Discover", href: "/discover", icon: <Compass className="h-5 w-5" /> },
      { title: "Downloaded", href: "/downloaded", icon: <Cpu className="h-5 w-5" /> },
      { title: "Adapters", href: "/adapters", icon: <Layers className="h-5 w-5" /> },
    ],
  },
  {
    label: "Workspace",
    items: [
      { title: "Chat", href: "/chat", icon: <MessageSquare className="h-5 w-5" /> },
      { title: "Train", href: "/train", icon: <Dumbbell className="h-5 w-5" /> },
      { title: "Evaluate", href: "/evaluate", icon: <Gauge className="h-5 w-5" /> },
    ],
  },
  {
    label: "System",
    items: [
      { title: "Environments", href: "/environments", icon: <Boxes className="h-5 w-5" /> },
      { title: "Serve", href: "/serve", icon: <Server className="h-5 w-5" /> },
      { title: "Settings", href: "/settings", icon: <Settings className="h-5 w-5" /> },
    ],
  },
];
