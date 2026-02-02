// DownloadedView.swift
// Local model management

import SwiftUI

struct DownloadedView: View {
    @EnvironmentObject var appState: AppState
    @State private var models: [LocalModel] = []
    @State private var isLoading = false
    @State private var selectedModel: LocalModel?
    @State private var showingDeleteConfirmation = false
    @State private var modelToDelete: LocalModel?
    
    private let columns = [
        GridItem(.adaptive(minimum: 280), spacing: 16)
    ]
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Downloaded Models")
                        .font(.title2.bold())
                    Text("\(models.count) models • \(totalSize) total")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: { Task { await loadModels() } }) {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(isLoading)
            }
            .padding()
            .background(.ultraThinMaterial)
            
            // Content
            if isLoading {
                Spacer()
                ProgressView()
                    .controlSize(.large)
                Spacer()
            } else if models.isEmpty {
                EmptyStateView(
                    icon: "square.stack.3d.up.slash",
                    title: "No Models Downloaded",
                    message: "Discover and pull models from HuggingFace to get started",
                    actionTitle: "Browse Models",
                    action: {
                        appState.selectedTab = .discover
                    }
                )
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 16) {
                        ForEach(models) { model in
                            DownloadedModelCard(
                                model: model,
                                onChat: { startChat(with: model) },
                                onTrain: { startTraining(with: model) },
                                onServe: { startServer(with: model) },
                                onDelete: { confirmDelete(model) }
                            )
                        }
                    }
                    .padding()
                }
            }
        }
        .navigationTitle("Downloaded")
        .toolbar {
            ToolbarItem {
                Menu {
                    Button("Import from File...") {
                        // Import logic
                    }
                    Button("Open Models Folder") {
                        NSWorkspace.shared.open(URL(fileURLWithPath: appState.projectRoot + "/models"))
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .alert("Delete Model?", isPresented: $showingDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                Task {
                    await deleteModel(model)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete '\(model.name)'? This action cannot be undone.")
        }
        .task {
            await loadModels()
        }
    }
    
    private var totalSize: String {
        let total = models.reduce(0) { $0 + $1.size }
        return ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
    }
    
    private func loadModels() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            models = try await appState.apiClient.listLocalModels()
        } catch {
            // Keep mock data on error
        }
    }
    
    private func startChat(with model: LocalModel) {
        // Set selected model in chat state
        appState.selectedTab = .chat
    }
    
    private func startTraining(with model: LocalModel) {
        appState.selectedTab = .train
    }
    
    private func startServer(with model: LocalModel) {
        appState.selectedTab = .serve
    }
    
    private func confirmDelete(_ model: LocalModel) {
        modelToDelete = model
        showingDeleteConfirmation = true
    }
    
    private func deleteModel(_ model: LocalModel) async {
        do {
            try await appState.apiClient.deleteModel(modelId: model.id)
            await loadModels()
        } catch {
            // Show error
        }
    }
}

// MARK: - Downloaded Model Card

struct DownloadedModelCard: View {
    let model: LocalModel
    let onChat: () -> Void
    let onTrain: () -> Void
    let onServe: () -> Void
    let onDelete: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(model.isAdapter ? Color.purple.opacity(0.2) : Color.blue.opacity(0.2))
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: model.isAdapter ? "wand.and.stars" : "cpu")
                        .font(.title3)
                        .foregroundColor(model.isAdapter ? .purple : .blue)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.name)
                        .font(.headline)
                        .lineLimit(1)
                    
                    HStack(spacing: 4) {
                        if model.isAdapter {
                            Text("Adapter")
                                .font(.caption2)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.purple.opacity(0.2))
                                .foregroundColor(.purple)
                                .cornerRadius(2)
                        }
                        
                        Text(model.sizeDisplay)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
            }
            
            Divider()
            
            // Metadata
            VStack(alignment: .leading, spacing: 4) {
                InfoRow(label: "Quantization", value: model.quantization)
                
                if let meta = model.metadata {
                    if let base = meta.baseModel {
                        InfoRow(label: "Base model", value: base)
                    }
                    if let arch = meta.architecture {
                        InfoRow(label: "Architecture", value: arch)
                    }
                }
                
                if let lastUsed = model.lastUsed {
                    InfoRow(
                        label: "Last used",
                        value: Formatters.relativeDateFormatter.localizedString(for: lastUsed, relativeTo: Date())
                    )
                } else {
                    InfoRow(label: "Last used", value: "Never")
                }
            }
            .font(.caption)
            
            Divider()
            
            // Actions
            HStack(spacing: 8) {
                ActionButton(icon: "bubble.left", label: "Chat", action: onChat)
                ActionButton(icon: "figure.strengthtraining.traditional", label: "Train", action: onTrain)
                ActionButton(icon: "server.rack", label: "Serve", action: onServe)
                
                Spacer()
                
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .help("Delete model")
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Info Row

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .lineLimit(1)
        }
    }
}

// MARK: - Action Button

struct ActionButton: View {
    let icon: String
    let label: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                Text(label)
                    .font(.caption2)
            }
            .frame(width: 50)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderless)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(6)
    }
}

// MARK: - Preview

struct DownloadedView_Previews: PreviewProvider {
    static var previews: some View {
        DownloadedView()
            .environmentObject(AppState())
    }
}
