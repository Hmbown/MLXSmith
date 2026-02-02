// ServeView.swift
// API server control and logs

import SwiftUI

struct ServeView: View {
    @EnvironmentObject var appState: AppState
    
    @State private var isServerRunning = false
    @State private var selectedModel = ""
    @State private var port: Int = 8080
    @State private var logs: [ServerLog] = []
    @State private var availableModels: [String] = []
    @State private var pulse = false
    @State private var showingClearConfirmation = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 16) {
                // Status indicator
                HStack(spacing: 8) {
                    Circle()
                        .fill(isServerRunning ? Color.green : Color.secondary)
                        .frame(width: 12, height: 12)
                        .overlay(
                            Circle()
                                .stroke(isServerRunning ? Color.green : Color.clear, lineWidth: 2)
                                .scaleEffect(pulse ? 1.5 : 1.0)
                                .opacity(pulse ? 0 : 0.5)
                        )
                        .animation(.easeInOut(duration: 1).repeatForever(autoreverses: false), value: pulse)
                    
                    Text(isServerRunning ? "Running" : "Stopped")
                        .font(.headline)
                }
                
                Divider()
                    .frame(height: 24)
                
                // Model selector
                Picker("Model:", selection: $selectedModel) {
                    Text("Select model...").tag("")
                    ForEach(availableModels, id: \.self) { model in
                        Text(model).tag(model)
                    }
                }
                .frame(width: 250)
                .disabled(isServerRunning)
                
                // Port field
                HStack {
                    Text("Port:")
                        .foregroundColor(.secondary)
                    TextField("Port", value: $port, format: .number)
                        .frame(width: 60)
                        .disabled(isServerRunning)
                }
                
                Spacer()
                
                // Endpoint info
                if isServerRunning {
                    Button(action: copyEndpoint) {
                        Label("http://localhost:\(port)", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                }
                
                // Toggle button
                Button(action: toggleServer) {
                    Label(
                        isServerRunning ? "Stop Server" : "Start Server",
                        systemImage: isServerRunning ? "stop.fill" : "play.fill"
                    )
                }
                .buttonStyle(isServerRunning ? .bordered : .borderedProminent)
                .tint(isServerRunning ? .red : .green)
                .disabled(selectedModel.isEmpty && !isServerRunning)
            }
            .padding()
            .background(.ultraThinMaterial)
            
            // Logs area
            VStack(spacing: 0) {
                HStack {
                    Text("Request Logs")
                        .font(.headline)
                    
                    Spacer()
                    
                    Text("\(logs.count) entries")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Button(action: { showingClearConfirmation = true }) {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .disabled(logs.isEmpty)
                }
                .padding()
                
                Divider()
                
                // Log list
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(logs) { log in
                            LogEntryRow(log: log)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .font(.system(size: 12, design: .monospaced))
            }
            
            // Endpoints reference
            VStack(alignment: .leading, spacing: 12) {
                Text("API Endpoints")
                    .font(.headline)
                
                HStack(spacing: 24) {
                    EndpointItem(method: "GET", path: "/health", description: "Health check")
                    EndpointItem(method: "POST", path: "/v1/chat/completions", description: "Chat completions")
                    EndpointItem(method: "POST", path: "/internal/rollout", description: "Generate with logprobs")
                }
                
                HStack(spacing: 24) {
                    EndpointItem(method: "GET", path: "/internal/rlm/state", description: "RLM state")
                    EndpointItem(method: "GET", path: "/internal/rlm/history", description: "RLM history")
                    EndpointItem(method: "POST", path: "/internal/adapter/reload", description: "Reload adapter")
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
        }
        .navigationTitle("Serve")
        .alert("Clear Logs?", isPresented: $showingClearConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Clear", role: .destructive) {
                logs.removeAll()
            }
        } message: {
            Text("This will clear all server logs. This action cannot be undone.")
        }
        .task {
            await loadModels()
            // Add some mock logs
            addMockLogs()
        }
        .onChange(of: isServerRunning) { running in
            if running {
                withAnimation(.easeInOut(duration: 1).repeatForever(autoreverses: false)) {
                    pulse = true
                }
            } else {
                pulse = false
            }
        }
    }
    
    private func loadModels() async {
        do {
            let models = try await appState.apiClient.listLocalModels()
            availableModels = models.map { $0.name }
            if !availableModels.isEmpty && selectedModel.isEmpty {
                selectedModel = availableModels[0]
            }
        } catch {
            availableModels = ["Llama 3.2 3B (4-bit)", "Qwen 2.5 7B (4-bit)"]
            selectedModel = availableModels[0]
        }
    }
    
    private func toggleServer() {
        if isServerRunning {
            appState.serverProcess?.stopServer()
            isServerRunning = false
            addLog("Server stopped by user", level: .info)
        } else {
            Task {
                do {
                    let manager = ServerProcessManager()
                    appState.serverProcess = manager
                    try await manager.startServer(
                        model: selectedModel,
                        port: port,
                        projectRoot: appState.projectRoot
                    )
                    isServerRunning = true
                    addLog("Server started on port \(port)", level: .info)
                } catch {
                    addLog("Failed to start server: \(error)", level: .error)
                }
            }
        }
    }
    
    private func copyEndpoint() {
        let endpoint = "http://localhost:\(port)"
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(endpoint, forType: .string)
    }
    
    private func addLog(_ message: String, level: ServerLog.LogLevel, endpoint: String? = nil) {
        let log = ServerLog(
            timestamp: Date(),
            level: level,
            message: message,
            endpoint: endpoint
        )
        logs.append(log)
        
        if logs.count > 1000 {
            logs.removeFirst(logs.count - 1000)
        }
    }
    
    private func addMockLogs() {
        let mockLogs = [
            ("Server ready on http://0.0.0.0:8080", ServerLog.LogLevel.info),
            ("GET /health 200 OK", ServerLog.LogLevel.request),
            ("POST /v1/chat/completions 200 OK - 1456ms", ServerLog.LogLevel.request),
            ("Generated 256 tokens in 1.45s (176 tok/s)", ServerLog.LogLevel.info),
            ("POST /v1/chat/completions 200 OK - 1234ms", ServerLog.LogLevel.request),
            ("GET /internal/rlm/state 200 OK", ServerLog.LogLevel.request),
        ]
        
        for (message, level) in mockLogs {
            addLog(message, level: level)
        }
    }
}

// MARK: - Log Entry Row

struct LogEntryRow: View {
    let log: ServerLog
    
    var body: some View {
        HStack(spacing: 8) {
            Text(timestampString)
                .foregroundColor(.secondary)
            
            Text(log.level.rawValue)
                .fontWeight(.bold)
                .foregroundColor(colorForLevel(log.level))
                .frame(width: 40, alignment: .leading)
            
            if let endpoint = log.endpoint {
                Text(endpoint)
                    .foregroundColor(.cyan)
            }
            
            Text(log.message)
                .foregroundColor(.primary)
            
            Spacer()
        }
    }
    
    private var timestampString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: log.timestamp)
    }
    
    private func colorForLevel(_ level: ServerLog.LogLevel) -> Color {
        switch level {
        case .info: return .secondary
        case .warning: return .yellow
        case .error: return .red
        case .request: return .cyan
        case .response: return .green
        }
    }
}

// MARK: - Endpoint Item

struct EndpointItem: View {
    let method: String
    let path: String
    let description: String
    
    var body: some View {
        HStack(spacing: 8) {
            Text(method)
                .font(.caption.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(methodColor)
                .foregroundColor(.white)
                .cornerRadius(3)
            
            VStack(alignment: .leading, spacing: 0) {
                Text(path)
                    .font(.caption.monospaced())
                Text(description)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private var methodColor: Color {
        switch method {
        case "GET": return .blue
        case "POST": return .green
        case "PUT": return .orange
        case "DELETE": return .red
        default: return .gray
        }
    }
}

// MARK: - Preview

struct ServeView_Previews: PreviewProvider {
    static var previews: some View {
        ServeView()
            .environmentObject(AppState())
    }
}
