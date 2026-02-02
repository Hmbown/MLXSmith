// SettingsView.swift
// App configuration and preferences

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var hfToken: String = ""
    @State private var showingToken = false
    @State private var testConnectionResult: Bool?
    @State private var isTesting = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // HuggingFace Token
                SectionCard(title: "HuggingFace Token") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Store your HuggingFace access token securely in Keychain")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        HStack {
                            if showingToken {
                                TextField("Token", text: $hfToken)
                                    .textFieldStyle(.roundedBorder)
                            } else {
                                SecureField("Token", text: $hfToken)
                                    .textFieldStyle(.roundedBorder)
                            }
                            
                            Button(action: { showingToken.toggle() }) {
                                Image(systemName: showingToken ? "eye.slash" : "eye")
                            }
                            .buttonStyle(.borderless)
                            
                            Button("Save") {
                                saveToken()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(hfToken.isEmpty)
                        }
                        
                        if let token = appState.keychain.hfToken {
                            HStack {
                                Image(systemName: "checkmark.shield.fill")
                                    .foregroundColor(.green)
                                Text("Token saved in Keychain")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                Spacer()
                                
                                Button("Delete") {
                                    deleteToken()
                                }
                                .buttonStyle(.borderless)
                                .foregroundColor(.red)
                                .controlSize(.small)
                            }
                            .padding(.top, 4)
                        }
                        
                        Link("Get your token from HuggingFace",
                             destination: URL(string: "https://huggingface.co/settings/tokens")!)
                            .font(.caption)
                    }
                }
                
                // Server Settings
                SectionCard(title: "Server Settings") {
                    VStack(spacing: 16) {
                        HStack {
                            Text("Default Host")
                            Spacer()
                            TextField("Host", text: $appState.serverHost)
                                .frame(width: 200)
                                .textFieldStyle(.roundedBorder)
                        }
                        
                        HStack {
                            Text("Default Port")
                            Spacer()
                            TextField("Port", value: $appState.serverPort, format: .number)
                                .frame(width: 200)
                                .textFieldStyle(.roundedBorder)
                        }
                        
                        HStack {
                            Spacer()
                            
                            if let result = testConnectionResult {
                                HStack {
                                    Image(systemName: result ? "checkmark.circle.fill" : "xmark.circle.fill")
                                        .foregroundColor(result ? .green : .red)
                                    Text(result ? "Connected" : "Connection failed")
                                        .foregroundColor(result ? .green : .red)
                                }
                            }
                            
                            Button(isTesting ? "Testing..." : "Test Connection") {
                                Task { await testConnection() }
                            }
                            .disabled(isTesting)
                            .controlSize(.small)
                        }
                    }
                }
                
                // Paths
                SectionCard(title: "Paths") {
                    VStack(spacing: 16) {
                        PathPicker(
                            label: "Project Root",
                            path: $appState.projectRoot,
                            prompt: "Select project root directory"
                        )
                        
                        PathPicker(
                            label: "Cache Directory",
                            path: $appState.cachePath,
                            prompt: "Select cache directory"
                        )
                        
                        HStack {
                            Spacer()
                            Button("Open Project Folder") {
                                NSWorkspace.shared.open(URL(fileURLWithPath: appState.projectRoot))
                            }
                            .controlSize(.small)
                        }
                    }
                }
                
                // Appearance
                SectionCard(title: "Appearance") {
                    Picker("Theme", selection: .constant(0)) {
                        Text("System").tag(0)
                        Text("Light").tag(1)
                        Text("Dark").tag(2)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 250)
                }
                
                // About
                SectionCard(title: "About MLXSmith") {
                    HStack(spacing: 16) {
                        Image(systemName: "cpu")
                            .font(.system(size: 48))
                            .foregroundColor(.accentColor)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("MLXSmith")
                                .font(.headline)
                            Text("Version 1.0.0")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Native macOS interface for MLX model training and inference")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            HStack(spacing: 16) {
                                Link("GitHub", destination: URL(string: "https://github.com")!)
                                Link("Documentation", destination: URL(string: "https://docs.mlxsmith.dev")!)
                            }
                            .font(.caption)
                            .padding(.top, 4)
                        }
                    }
                }
            }
            .padding()
        }
        .frame(minWidth: 500)
    }
    
    private func saveToken() {
        appState.keychain.hfToken = hfToken
        hfToken = ""
    }
    
    private func deleteToken() {
        appState.keychain.hfToken = nil
    }
    
    private func testConnection() async {
        isTesting = true
        defer { isTesting = false }
        
        testConnectionResult = await appState.apiClient.checkHealth()
        
        // Clear result after 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            testConnectionResult = nil
        }
    }
}

// MARK: - Path Picker

struct PathPicker: View {
    let label: String
    @Binding var path: String
    let prompt: String
    
    var body: some View {
        HStack {
            Text(label)
            Spacer()
            
            Text(path)
                .font(.caption.monospaced())
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 200, alignment: .trailing)
            
            Button("Browse...") {
                selectDirectory()
            }
            .controlSize(.small)
        }
    }
    
    private func selectDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = prompt
        panel.prompt = "Select"
        
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

// MARK: - Section Card

struct SectionCard<Content: View>: View {
    let title: String
    let content: Content
    
    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            
            content
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
    }
}

// MARK: - Preview

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
            .environmentObject(AppState())
    }
}
