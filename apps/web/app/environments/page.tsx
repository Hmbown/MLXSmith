"use client";

import { useState } from "react";
import { Boxes, Copy, Check, Terminal, Globe, ShieldCheck, Box } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusPill } from "@/components/ui/status-pill";
import { CommandCard } from "@/components/ui/command-card";
import { IconBadge } from "@/components/ui/icon-badge";

interface EnvironmentItem {
  id: string;
  name: string;
  description: string;
  icon: JSX.Element;
  primitives: string[];
  requiresDocker: boolean;
  installed: boolean;
}

const environments: EnvironmentItem[] = [
  {
    id: "code-sandbox",
    name: "Code Sandbox",
    description: "Python + shell execution for code tasks",
    icon: <ShieldCheck className="h-4 w-4" />,
    primitives: ["python()", "bash()", "file_io()", "network()"],
    requiresDocker: false,
    installed: true,
  },
  {
    id: "python-sandbox",
    name: "Python Sandbox",
    description: "Isolated Python worker with MLX",
    icon: <Terminal className="h-4 w-4" />,
    primitives: ["python()", "file_io()"],
    requiresDocker: false,
    installed: true,
  },
  {
    id: "docker-sandbox",
    name: "Docker Sandbox",
    description: "Containerized environment for verifier runs",
    icon: <Box className="h-4 w-4" />,
    primitives: ["bash()", "python()", "network()"],
    requiresDocker: true,
    installed: false,
  },
  {
    id: "web-browser",
    name: "Web Browser",
    description: "Headless browser for web navigation tasks",
    icon: <Globe className="h-4 w-4" />,
    primitives: ["navigate()", "click()", "extract()", "screenshot()"],
    requiresDocker: true,
    installed: false,
  },
  {
    id: "terminal",
    name: "Terminal Shell",
    description: "Unix shell for system commands",
    icon: <Terminal className="h-4 w-4" />,
    primitives: ["exec()", "cd()", "env()", "pipe()"],
    requiresDocker: false,
    installed: true,
  },
];

export default function EnvironmentsPage() {
  const [copied, setCopied] = useState<string | null>(null);

  const copy = async (command: string, key: string) => {
    await navigator.clipboard.writeText(command);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center justify-between border-b border-border/60 px-6">
        <div className="flex items-center gap-4">
          <Boxes className="h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-lg font-semibold">Environments</h1>
            <p className="text-sm text-muted-foreground">RFT-ready execution environments</p>
          </div>
        </div>
        <StatusPill tone="blue" dot={false}>
          CLI-managed
        </StatusPill>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Environment Registry</CardTitle>
              <CardDescription>Use the CLI to install or inspect environments.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2 text-sm">
              <Button
                variant="outline"
                size="sm"
                onClick={() => copy("mlxsmith env list", "list")}
              >
                {copied === "list" ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                List Environments
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => copy("mlxsmith env registry", "registry")}
              >
                {copied === "registry" ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                Show Registry
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => copy("mlxsmith env init myenv", "init")}
              >
                {copied === "init" ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                Scaffold New Env
              </Button>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {environments.map((env) => (
              <Card key={env.id} className="flex h-full flex-col">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <IconBadge tone={env.requiresDocker ? "yellow" : "green"}>
                      {env.icon}
                    </IconBadge>
                    {env.name}
                  </CardTitle>
                  <CardDescription>{env.description}</CardDescription>
                </CardHeader>
                <CardContent className="mt-auto space-y-3 text-sm text-muted-foreground">
                  <div className="flex items-center justify-between">
                    <span>Status</span>
                    <StatusPill tone={env.installed ? "green" : "yellow"} dot={false}>
                      {env.installed ? "Installed" : "Not installed"}
                    </StatusPill>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-muted-foreground">Primitives</div>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {env.primitives.map((primitive) => (
                        <span
                          key={primitive}
                          className="rounded-full border border-border/60 bg-muted/50 px-2 py-1 text-xs"
                        >
                          {primitive}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copy(`mlxsmith env info ${env.id}`, `${env.id}-info`)}
                    >
                      {copied === `${env.id}-info` ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                      Info
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copy(`mlxsmith env install ${env.id}`, `${env.id}-install`)}
                    >
                      {copied === `${env.id}-install` ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                      Install
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copy(`mlxsmith env run ${env.id} --model <adapter>`, `${env.id}-run`)}
                    >
                      {copied === `${env.id}-run` ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                      Run
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <CommandCard
            title="Package and Publish"
            description="Bundle an environment and publish to the registry."
            command="mlxsmith env package myenv && mlxsmith env publish envs/packages/myenv-0.1.0.tar.gz"
          />
        </div>
      </ScrollArea>
    </div>
  );
}
