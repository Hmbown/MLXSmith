"use client";

import { useState } from "react";
import { Settings, Key, Folder, Globe, Moon, Sun, Monitor, Check, Loader2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTheme } from "next-themes";
import { useSettingsStore } from "@/stores/settings";
import { useHFToken } from "@/hooks/useSettings";
import { StatusPill } from "@/components/ui/status-pill";
import { useServerStatus } from "@/hooks/useServer";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { apiUrl, setApiUrl, projectPath, setProjectPath, defaultModel, defaultTemperature, defaultMaxTokens } =
    useSettingsStore();
  const { setToken, isLoading: hfTokenLoading } = useHFToken();
  const { data: serverStatus } = useServerStatus();

  const [localApiUrl, setLocalApiUrl] = useState(apiUrl);
  const [localProjectPath, setLocalProjectPath] = useState(projectPath || "");
  const [localHfToken, setLocalHfToken] = useState("");
  const [saved, setSaved] = useState(false);

  const showSaved = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleSaveApiUrl = () => {
    setApiUrl(localApiUrl);
    showSaved();
  };

  const handleSaveProjectPath = () => {
    setProjectPath(localProjectPath);
    showSaved();
  };

  const handleSaveHFToken = async () => {
    if (!localHfToken.trim()) return;
    try {
      await setToken(localHfToken.trim());
      setLocalHfToken("");
      showSaved();
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Failed to save token");
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-16 items-center border-b border-border/60 px-6">
        <Settings className="mr-2 h-5 w-5 text-muted-foreground" />
        <div>
          <h1 className="text-lg font-semibold">Settings</h1>
          <p className="text-sm text-muted-foreground">Configure MLXSmith preferences</p>
        </div>
      </header>

      <ScrollArea className="flex-1 p-6">
        <div className="mx-auto max-w-3xl space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {theme === "dark" ? (
                  <Moon className="h-5 w-5" />
                ) : theme === "light" ? (
                  <Sun className="h-5 w-5" />
                ) : (
                  <Monitor className="h-5 w-5" />
                )}
                Appearance
              </CardTitle>
              <CardDescription>Choose your preferred theme</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={theme} onValueChange={(v) => setTheme(v)}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="light" className="gap-2">
                    <Sun className="h-4 w-4" />
                    Light
                  </TabsTrigger>
                  <TabsTrigger value="dark" className="gap-2">
                    <Moon className="h-4 w-4" />
                    Dark
                  </TabsTrigger>
                  <TabsTrigger value="system" className="gap-2">
                    <Monitor className="h-4 w-4" />
                    System
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  API Configuration
                </CardTitle>
                <CardDescription>Configure the MLXSmith API endpoint</CardDescription>
              </div>
              <StatusPill tone={serverStatus?.running ? "green" : "red"} dot={false}>
                {serverStatus?.running ? "Online" : "Offline"}
              </StatusPill>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-url">API URL</Label>
                <div className="flex gap-2">
                  <Input
                    id="api-url"
                    value={localApiUrl}
                    onChange={(e) => setLocalApiUrl(e.target.value)}
                    placeholder="http://localhost:8080"
                  />
                  <Button onClick={handleSaveApiUrl}>
                    {saved ? <Check className="h-4 w-4" /> : "Save"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  The URL of your MLXSmith API server (used for chat + models).
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                HuggingFace Token
              </CardTitle>
              <CardDescription>Store your HF token on the API server.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="hf-token">Access Token</Label>
                <div className="flex gap-2">
                  <Input
                    id="hf-token"
                    type="password"
                    value={localHfToken}
                    onChange={(e) => setLocalHfToken(e.target.value)}
                    placeholder="hf_..."
                  />
                  <Button onClick={handleSaveHFToken} disabled={hfTokenLoading || !localHfToken.trim()}>
                    {hfTokenLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  The token is stored securely server-side. Re-enter to update.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Folder className="h-5 w-5" />
                Project Path
              </CardTitle>
              <CardDescription>Default location for models, adapters, and data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="project-path">Project Root</Label>
                <div className="flex gap-2">
                  <Input
                    id="project-path"
                    value={localProjectPath}
                    onChange={(e) => setLocalProjectPath(e.target.value)}
                    placeholder="/path/to/project"
                  />
                  <Button onClick={handleSaveProjectPath}>
                    {saved ? <Check className="h-4 w-4" /> : "Save"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Used for CLI command defaults in the web UI.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Default Chat Settings</CardTitle>
              <CardDescription>Defaults for new chat sessions</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Temperature</Label>
                <p className="font-medium">{defaultTemperature}</p>
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">Max Tokens</Label>
                <p className="font-medium">{defaultMaxTokens}</p>
              </div>
              <div className="space-y-1 sm:col-span-2">
                <Label className="text-xs text-muted-foreground">Default Model</Label>
                <p className="font-medium">{defaultModel || "Not set"}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>About</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <span className="font-medium text-foreground">MLXSmith</span> - Train, serve, and chat with MLX models
                </p>
                <p>Version 0.1.0</p>
                <p>Web console aligned with the MLXSmith CLI.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  );
}
