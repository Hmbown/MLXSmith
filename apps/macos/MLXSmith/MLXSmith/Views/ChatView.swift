// ChatView.swift
// Native chat interface for inference

import SwiftUI

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var availableModels: [String] = []
    @State private var isGenerating = false
    @State private var showParameters = true
    @State private var systemPrompt = "You are a helpful AI assistant."
    
    // Parameters
    @State private var temperature: Double = 0.7
    @State private var maxTokens: Double = 256
    @State private var topP: Double = 1.0
    @State private var topK: Double = 40
    
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        HStack(spacing: 0) {
            // Chat area
            VStack(spacing: 0) {
                // Header
                HStack {
                    Picker("Model", selection: $selectedModel) {
                        Text("Select model...").tag("")
                        ForEach(availableModels, id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                    .frame(width: 250)
                    .disabled(isGenerating)
                    
                    Spacer()
                    
                    if isGenerating {
                        HStack(spacing: 6) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Generating...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Button(action: { showParameters.toggle() }) {
                        Image(systemName: showParameters ? "sidebar.right" : "sidebar.left")
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
                
                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(messages) { message in
                                MessageBubble(message: message)
                            }
                            if isGenerating && messages.last?.role != .assistant {
                                TypingIndicator()
                            }
                        }
                        .padding()
                        .id("bottom")
                    }
                    .onChange(of: messages) { _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
                
                Divider()
                
                // Input area
                VStack(spacing: 8) {
                    HStack(spacing: 12) {
                        TextEditor(text: $inputText)
                            .font(.body)
                            .frame(minHeight: 40, maxHeight: 120)
                            .focused($isInputFocused)
                            .scrollContentBackground(.hidden)
                            .onKeyPress(.return, modifiers: []) {
                                if !inputText.isEmpty && !isGenerating {
                                    sendMessage()
                                    return .handled
                                }
                                return .ignored
                            }
                        
                        VStack(spacing: 8) {
                            Button(action: sendMessage) {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(inputText.isEmpty || isGenerating ? .secondary : .accentColor)
                            }
                            .disabled(inputText.isEmpty || isGenerating)
                            .keyboardShortcut(.return, modifiers: [.command])
                            
                            Button(action: clearChat) {
                                Image(systemName: "xmark.circle")
                                    .font(.title3)
                                    .foregroundColor(.secondary)
                            }
                            .disabled(messages.isEmpty)
                            .help("Clear chat")
                        }
                    }
                    
                    HStack {
                        Text("⌘↵ to send")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        if let lastMessage = messages.last,
                           let tps = lastMessage.tokensPerSecond {
                            Text("\(Formatters.formatTokensPerSecond(tps))")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
            }
            
            // Parameters sidebar
            if showParameters {
                Divider()
                
                VStack(alignment: .leading, spacing: 16) {
                    Text("Parameters")
                        .font(.headline)
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            // System Prompt
                            VStack(alignment: .leading, spacing: 8) {
                                Text("System Prompt")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                TextEditor(text: $systemPrompt)
                                    .font(.caption)
                                    .frame(height: 80)
                                    .padding(4)
                                    .background(Color(nsColor: .textBackgroundColor))
                                    .cornerRadius(6)
                            }
                            
                            Divider()
                            
                            // Temperature
                            ParameterSlider(
                                label: "Temperature",
                                value: $temperature,
                                range: 0...2,
                                step: 0.1
                            )
                            
                            // Max Tokens
                            ParameterSlider(
                                label: "Max Tokens",
                                value: $maxTokens,
                                range: 64...4096,
                                step: 64,
                                format: { "\(Int($0))" }
                            )
                            
                            // Top P
                            ParameterSlider(
                                label: "Top P",
                                value: $topP,
                                range: 0...1,
                                step: 0.05
                            )
                            
                            // Top K
                            ParameterSlider(
                                label: "Top K",
                                value: $topK,
                                range: 1...100,
                                step: 1,
                                format: { "\(Int($0))" }
                            )
                            
                            Divider()
                            
                            // Reset button
                            Button("Reset to Defaults") {
                                resetParameters()
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                    
                    Spacer()
                }
                .padding()
                .frame(width: 220)
                .background(Color(nsColor: .controlBackgroundColor))
            }
        }
        .navigationTitle("Chat")
        .toolbar {
            ToolbarItem {
                Button(action: exportChat) {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(messages.isEmpty)
            }
        }
        .task {
            await loadModels()
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
            // Use mock data
            availableModels = ["Llama 3.2 3B (4-bit)", "Qwen 2.5 7B (4-bit)"]
            selectedModel = availableModels[0]
        }
    }
    
    private func sendMessage() {
        guard !inputText.isEmpty && !isGenerating else { return }
        
        let userMessage = ChatMessage(role: .user, content: inputText)
        messages.append(userMessage)
        
        let currentInput = inputText
        inputText = ""
        isGenerating = true
        
        Task {
            await generateResponse(for: currentInput)
        }
    }
    
    private func generateResponse(for userInput: String) async {
        var chatMessages = [ChatMessage(role: .system, content: systemPrompt)]
        chatMessages.append(contentsOf: messages.filter { $0.role != .system })
        
        let startTime = Date()
        var tokenCount = 0
        var responseText = ""
        
        do {
            let stream = try await appState.apiClient.sendChatMessage(
                messages: chatMessages,
                model: selectedModel,
                maxTokens: Int(maxTokens),
                temperature: temperature,
                topP: topP,
                topK: Int(topK),
                stream: true
            )
            
            for try await chunk in stream {
                responseText += chunk
                tokenCount += 1
                
                if let lastIndex = messages.indices.last,
                   messages[lastIndex].role == .assistant {
                    messages[lastIndex].content = responseText
                } else {
                    let assistantMessage = ChatMessage(
                        role: .assistant,
                        content: responseText,
                        tokensPerSecond: nil
                    )
                    messages.append(assistantMessage)
                }
            }
            
            // Update with final stats
            let duration = Date().timeIntervalSince(startTime)
            let tps = duration > 0 ? Double(tokenCount) / duration : 0
            
            if let lastIndex = messages.indices.last {
                messages[lastIndex].tokensPerSecond = tps
                messages[lastIndex].totalTokens = tokenCount
            }
            
        } catch {
            let errorMessage = ChatMessage(
                role: .assistant,
                content: "Error: \(error.localizedDescription)"
            )
            messages.append(errorMessage)
        }
        
        isGenerating = false
    }
    
    private func clearChat() {
        messages.removeAll()
    }
    
    private func resetParameters() {
        temperature = 0.7
        maxTokens = 256
        topP = 1.0
        topK = 40
        systemPrompt = "You are a helpful AI assistant."
    }
    
    private func exportChat() {
        // Export conversation to file
        let conversation = messages.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n\n")
        // Present save panel
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.role == .assistant {
                Spacer(minLength: 60)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: iconForRole(message.role))
                        .foregroundColor(colorForRole(message.role))
                    Text(message.role.rawValue.capitalized)
                        .font(.caption.bold())
                        .foregroundColor(colorForRole(message.role))
                    
                    Spacer()
                    
                    Text(message.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Text(message.content)
                    .font(.body)
                    .textSelection(.enabled)
                    .padding(.top, 2)
                
                if let tps = message.tokensPerSecond {
                    HStack {
                        Spacer()
                        Text("\(Formatters.formatTokensPerSecond(tps))")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(12)
            .background(backgroundForRole(message.role))
            .cornerRadius(12)
            
            if message.role == .user {
                Spacer(minLength: 60)
            }
        }
    }
    
    private func iconForRole(_ role: ChatMessage.MessageRole) -> String {
        switch role {
        case .system: return "gearshape"
        case .user: return "person"
        case .assistant: return "cpu"
        }
    }
    
    private func colorForRole(_ role: ChatMessage.MessageRole) -> Color {
        switch role {
        case .system: return .secondary
        case .user: return .blue
        case .assistant: return .green
        }
    }
    
    private func backgroundForRole(_ role: ChatMessage.MessageRole) -> Color {
        switch role {
        case .system:
            return Color.secondary.opacity(0.1)
        case .user:
            return Color.blue.opacity(0.15)
        case .assistant:
            return Color(nsColor: .controlBackgroundColor)
        }
    }
}

// MARK: - Typing Indicator

struct TypingIndicator: View {
    @State private var offset: CGFloat = 0
    
    var body: some View {
        HStack {
            Spacer(minLength: 60)
            
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle()
                        .frame(width: 6, height: 6)
                        .offset(y: offset)
                        .animation(
                            .easeInOut(duration: 0.5)
                            .repeatForever(autoreverses: true)
                            .delay(Double(i) * 0.15),
                            value: offset
                        )
                }
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(12)
        }
        .onAppear {
            offset = -4
        }
    }
}

// MARK: - Parameter Slider

struct ParameterSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var format: (Double) -> String = { String(format: "%.2f", $0) }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption)
                Spacer()
                Text(format(value))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            
            Slider(value: $value, in: range, step: step)
        }
    }
}

// MARK: - Preview

struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        ChatView()
            .environmentObject(AppState())
    }
}
