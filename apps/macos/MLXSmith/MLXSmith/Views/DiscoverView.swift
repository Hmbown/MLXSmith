// DiscoverView.swift
// HuggingFace model browser

import SwiftUI

struct DiscoverView: View {
    @StateObject private var hfAPI = HuggingFaceAPI()
    @EnvironmentObject var appState: AppState
    
    @State private var searchQuery = ""
    @State private var selectedFilter: ModelFilter = .all
    @State private var selectedModel: HFModel?
    @State private var showingPullSheet = false
    @State private var pullInProgress = false
    @State private var pullProgress: Double = 0
    
    enum ModelFilter: String, CaseIterable {
        case all = "All"
        case mlx = "MLX"
        case textGeneration = "Text Gen"
        case embeddings = "Embeddings"
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Search and filter bar
            HStack(spacing: 12) {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Search HuggingFace models...", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .onSubmit {
                        Task {
                            await hfAPI.searchModels(query: searchQuery)
                        }
                    }
                
                if !searchQuery.isEmpty {
                    Button(action: { searchQuery = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                
                Divider()
                    .frame(height: 20)
                
                Picker("", selection: $selectedFilter) {
                    ForEach(ModelFilter.allCases, id: \.self) { filter in
                        Text(filter.rawValue).tag(filter)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 250)
            }
            .padding()
            .background(.ultraThinMaterial)
            
            // Content
            if hfAPI.isLoading {
                Spacer()
                ProgressView("Loading models...")
                    .controlSize(.large)
                Spacer()
            } else if hfAPI.models.isEmpty {
                EmptyStateView(
                    icon: "magnifyingglass",
                    title: "No Models Found",
                    message: "Try adjusting your search or browse trending models"
                )
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 300))], spacing: 16) {
                        ForEach(filteredModels) { model in
                            ModelCard(model: model) {
                                selectedModel = model
                                showingPullSheet = true
                            }
                        }
                    }
                    .padding()
                }
            }
        }
        .navigationTitle("Discover")
        .toolbar {
            ToolbarItem {
                Button(action: {
                    Task {
                        await hfAPI.fetchTrendingModels()
                    }
                }) {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(hfAPI.isLoading)
            }
        }
        .sheet(isPresented: $showingPullSheet) {
            if let model = selectedModel {
                PullModelSheet(
                    model: model,
                    isPresented: $showingPullSheet
                )
                .environmentObject(appState)
            }
        }
        .task {
            await hfAPI.fetchTrendingModels()
        }
    }
    
    private var filteredModels: [HFModel] {
        switch selectedFilter {
        case .all:
            return hfAPI.models
        case .mlx:
            return hfAPI.models.filter { $0.isMLXCompatible }
        case .textGeneration:
            return hfAPI.models.filter { $0.pipelineTag == "text-generation" }
        case .embeddings:
            return hfAPI.models.filter { $0.tags.contains("embeddings") }
        }
    }
}

// MARK: - Model Card

struct ModelCard: View {
    let model: HFModel
    let onPull: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                // Organization avatar placeholder
                ZStack {
                    Circle()
                        .fill(organizationColor)
                        .frame(width: 40, height: 40)
                    Text(String(model.organization.prefix(1)))
                        .font(.headline)
                        .foregroundColor(.white)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.organization)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(model.displayName)
                        .font(.headline)
                        .lineLimit(1)
                }
                
                Spacer()
                
                if model.isMLXCompatible {
                    MLXBadge()
                }
            }
            
            // Description
            if let description = model.description {
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            // Tags
            FlowLayout(spacing: 4) {
                ForEach(model.tags.prefix(4), id: \.self) { tag in
                    TagView(tag: tag)
                }
            }
            
            Divider()
            
            // Footer stats
            HStack {
                Label(Formatters.formatNumber(model.downloads), systemImage: "arrow.down.circle")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Label(Formatters.formatNumber(model.likes), systemImage: "heart")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(model.sizeEstimate)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Pull button
            Button(action: onPull) {
                Label("Pull to MLX", systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
        )
    }
    
    private var organizationColor: Color {
        let colors: [Color] = [.blue, .green, .orange, .purple, .pink, .teal]
        let hash = abs(model.organization.hashValue)
        return colors[hash % colors.count]
    }
}

// MARK: - MLX Badge

struct MLXBadge: View {
    var body: some View {
        Text("MLX")
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.green.opacity(0.2))
            .foregroundColor(.green)
            .cornerRadius(4)
    }
}

// MARK: - Tag View

struct TagView: View {
    let tag: String
    
    var body: some View {
        Text(tag)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.secondary.opacity(0.15))
            .foregroundColor(.secondary)
            .cornerRadius(4)
    }
}

// MARK: - Flow Layout

struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = FlowResult(in: proposal.width ?? 0, subviews: subviews, spacing: spacing)
        return result.size
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = FlowResult(in: bounds.width, subviews: subviews, spacing: spacing)
        for (index, subview) in subviews.enumerated() {
            subview.place(at: CGPoint(x: bounds.minX + result.positions[index].x,
                                      y: bounds.minY + result.positions[index].y),
                         proposal: .unspecified)
        }
    }
    
    struct FlowResult {
        var size: CGSize = .zero
        var positions: [CGPoint] = []
        
        init(in maxWidth: CGFloat, subviews: Subviews, spacing: CGFloat) {
            var x: CGFloat = 0
            var y: CGFloat = 0
            var rowHeight: CGFloat = 0
            
            for subview in subviews {
                let size = subview.sizeThatFits(.unspecified)
                
                if x + size.width > maxWidth && x > 0 {
                    x = 0
                    y += rowHeight + spacing
                    rowHeight = 0
                }
                
                positions.append(CGPoint(x: x, y: y))
                rowHeight = max(rowHeight, size.height)
                x += size.width + spacing
                
                self.size.width = max(self.size.width, x)
            }
            
            self.size.height = y + rowHeight
        }
    }
}

// MARK: - Pull Model Sheet

struct PullModelSheet: View {
    let model: HFModel
    @Binding var isPresented: Bool
    @EnvironmentObject var appState: AppState
    
    @State private var quantize = false
    @State private var qBits = 4
    @State private var isPulling = false
    @State private var progress: Double = 0
    @State private var statusMessage = ""
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            VStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)
                
                Text("Pull Model")
                    .font(.title2.bold())
                
                Text(model.modelId)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Divider()
            
            // Options
            VStack(alignment: .leading, spacing: 16) {
                Toggle("Quantize model", isOn: $quantize)
                
                if quantize {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Quantization bits: \(qBits)-bit")
                            .font(.caption)
                        
                        Picker("", selection: $qBits) {
                            Text("4-bit (recommended)").tag(4)
                            Text("6-bit").tag(6)
                            Text("8-bit").tag(8)
                        }
                        .pickerStyle(.segmented)
                    }
                }
                
                HStack {
                    Text("Estimated size:")
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(model.sizeEstimate)
                        .fontWeight(.medium)
                }
            }
            
            if isPulling {
                VStack(spacing: 8) {
                    ProgressView(value: progress, total: 1.0)
                        .progressViewStyle(.linear)
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // Buttons
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .disabled(isPulling)
                
                Spacer()
                
                Button(isPulling ? "Pulling..." : "Pull Model") {
                    Task {
                        await pullModel()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isPulling)
            }
        }
        .padding()
        .frame(width: 400, height: 350)
    }
    
    private func pullModel() async {
        isPulling = true
        statusMessage = "Downloading..."
        
        do {
            try await appState.apiClient.pullModel(
                modelId: model.modelId,
                quantize: quantize,
                qBits: quantize ? qBits : nil
            )
            
            // Simulate progress for now
            for i in 0...10 {
                try? await Task.sleep(nanoseconds: 300_000_000)
                progress = Double(i) / 10.0
            }
            
            statusMessage = "Complete!"
            try? await Task.sleep(nanoseconds: 500_000_000)
            isPresented = false
        } catch {
            statusMessage = "Error: \(error.localizedDescription)"
        }
        
        isPulling = false
    }
}

// MARK: - Preview

struct DiscoverView_Previews: PreviewProvider {
    static var previews: some View {
        DiscoverView()
            .environmentObject(AppState())
    }
}
