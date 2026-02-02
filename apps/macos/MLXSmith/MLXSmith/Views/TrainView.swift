// TrainView.swift
// RLM monitor and training interface

import SwiftUI
import Charts

struct TrainView: View {
    @EnvironmentObject var appState: AppState
    
    @State private var rlmState: RLMState?
    @State private var history: [RLMHistoryEntry] = []
    @State private var isLoading = false
    @State private var selectedTab: TrainTab = .monitor
    @State private var trainingConfig = TrainingConfig.default
    @State private var isTraining = false
    
    enum TrainTab: String, CaseIterable {
        case monitor = "Monitor"
        case config = "Configure"
        case datasets = "Datasets"
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Training")
                        .font(.title2.bold())
                    
                    if let state = rlmState {
                        HStack(spacing: 12) {
                            StatusBadge(
                                text: state.isTraining == true ? "Training" : "Idle",
                                color: state.isTraining == true ? .green : .secondary
                            )
                            
                            if let iter = state.lastIteration {
                                Text("Iteration \(iter)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            if let score = state.emaScore {
                                Text("EMA: \(String(format: "%.3f", score))")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                
                Spacer()
                
                Picker("", selection: $selectedTab) {
                    ForEach(TrainTab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)
                
                Button(isTraining ? "Stop" : "Start Training") {
                    toggleTraining()
                }
                .buttonStyle(isTraining ? .bordered : .borderedProminent)
                .tint(isTraining ? .red : .accentColor)
            }
            .padding()
            .background(.ultraThinMaterial)
            
            // Content
            Group {
                switch selectedTab {
                case .monitor:
                    MonitorTab(history: history, state: rlmState)
                case .config:
                    ConfigTab(config: $trainingConfig)
                case .datasets:
                    DatasetsTab()
                }
            }
        }
        .navigationTitle("Train")
        .task {
            await loadData()
        }
        .onReceive(Timer.publish(every: 5, on: .main, in: .common).autoconnect()) { _ in
            Task {
                await loadData()
            }
        }
    }
    
    private func loadData() async {
        do {
            rlmState = try await appState.apiClient.getRLMState()
            history = try await appState.apiClient.getRLMHistory()
        } catch {
            // Use mock data for preview
            rlmState = RLMState(
                lastIteration: 42,
                currentAdapter: "runs/rlm_001/adapter_v042",
                bestAdapter: "runs/rlm_001/adapter_v038",
                bestScore: 0.89,
                emaScore: 0.87,
                isTraining: true,
                currentStep: 420,
                totalSteps: 1000
            )
            
            // Generate mock history
            history = (0..<50).map { i in
                RLMHistoryEntry(
                    iteration: i,
                    adapterScore: 0.5 + Double(i) * 0.01 + Double.random(in: -0.05...0.05),
                    loss: 2.0 - Double(i) * 0.03 + Double.random(in: -0.1...0.1),
                    reward: 0.4 + Double(i) * 0.012 + Double.random(in: -0.05...0.05),
                    timestamp: Date().addingTimeInterval(Double(-50 + i) * 300)
                )
            }
        }
    }
    
    private func toggleTraining() {
        isTraining.toggle()
    }
}

// MARK: - Monitor Tab

struct MonitorTab: View {
    let history: [RLMHistoryEntry]
    let state: RLMState?
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Stats cards
                HStack(spacing: 16) {
                    StatCard(
                        title: "Best Score",
                        value: state?.bestScore.map { String(format: "%.3f", $0) } ?? "—",
                        icon: "star.fill",
                        color: .yellow
                    )
                    
                    StatCard(
                        title: "EMA Score",
                        value: state?.emaScore.map { String(format: "%.3f", $0) } ?? "—",
                        icon: "chart.line.uptrend.xyaxis",
                        color: .blue
                    )
                    
                    StatCard(
                        title: "Iteration",
                        value: state?.lastIteration.map { "\($0)" } ?? "—",
                        icon: "number",
                        color: .green
                    )
                    
                    if let current = state?.currentStep, let total = state?.totalSteps {
                        StatCard(
                            title: "Progress",
                            value: "\(current)/\(total)",
                            icon: "progress.indicator",
                            color: .purple
                        )
                    }
                }
                
                // Charts
                if !history.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Performance")
                            .font(.headline)
                        
                        Chart(history) { entry in
                            LineMark(
                                x: .value("Iteration", entry.iteration),
                                y: .value("Score", entry.adapterScore ?? 0)
                            )
                            .foregroundStyle(.blue)
                            
                            if let reward = entry.reward {
                                LineMark(
                                    x: .value("Iteration", entry.iteration),
                                    y: .value("Reward", reward)
                                )
                                .foregroundStyle(.green)
                            }
                        }
                        .frame(height: 200)
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Loss")
                            .font(.headline)
                        
                        Chart(history) { entry in
                            LineMark(
                                x: .value("Iteration", entry.iteration),
                                y: .value("Loss", entry.loss ?? 0)
                            )
                            .foregroundStyle(.red)
                        }
                        .frame(height: 150)
                        .chartYAxis {
                            AxisMarks(position: .leading)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                }
                
                // Current adapters
                if let state = state {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Current Adapters")
                            .font(.headline)
                        
                        HStack(spacing: 16) {
                            if let current = state.currentAdapter {
                                AdapterCard(
                                    label: "Current",
                                    path: current,
                                    isBest: false
                                )
                            }
                            
                            if let best = state.bestAdapter {
                                AdapterCard(
                                    label: "Best",
                                    path: best,
                                    isBest: true
                                )
                            }
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                }
            }
            .padding()
        }
    }
}

// MARK: - Config Tab

struct ConfigTab: View {
    @Binding var config: TrainingConfig
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Model section
                SectionCard(title: "Model") {
                    VStack(alignment: .leading, spacing: 12) {
                        TextField("Model ID", text: $config.modelId)
                        
                        HStack {
                            Text("Output Path")
                                .foregroundColor(.secondary)
                            Spacer()
                            TextField("", text: $config.outputPath)
                                .frame(width: 200)
                        }
                    }
                }
                
                // Training parameters
                SectionCard(title: "Training Parameters") {
                    VStack(spacing: 16) {
                        HStack {
                            Text("Learning Rate")
                            Spacer()
                            Text(String(format: "%.1e", config.learningRate))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $config.learningRate, in: 1e-5...1e-3, step: 1e-5)
                        
                        HStack {
                            Text("Batch Size")
                            Spacer()
                            Text("\(config.batchSize)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: .init(
                            get: { Double(config.batchSize) },
                            set: { config.batchSize = Int($0) }
                        ), in: 1...8, step: 1)
                        
                        HStack {
                            Text("Gradient Accumulation")
                            Spacer()
                            Text("\(config.gradientAccumulation)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: .init(
                            get: { Double(config.gradientAccumulation) },
                            set: { config.gradientAccumulation = Int($0) }
                        ), in: 1...32, step: 1)
                        
                        HStack {
                            Text("Iterations")
                            Spacer()
                            Text("\(config.iterations)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: .init(
                            get: { Double(config.iterations) },
                            set: { config.iterations = Int($0) }
                        ), in: 100...10000, step: 100)
                    }
                }
                
                // LoRA section
                SectionCard(title: "LoRA Configuration") {
                    VStack(spacing: 16) {
                        HStack {
                            Text("Rank (r)")
                            Spacer()
                            Text("\(config.loraRank)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: .init(
                            get: { Double(config.loraRank) },
                            set: { config.loraRank = Int($0) }
                        ), in: 4...128, step: 4)
                        
                        HStack {
                            Text("Alpha")
                            Spacer()
                            Text("\(config.loraAlpha)")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: .init(
                            get: { Double(config.loraAlpha) },
                            set: { config.loraAlpha = Int($0) }
                        ), in: 8...256, step: 8)
                        
                        HStack {
                            Text("Dropout")
                            Spacer()
                            Text(String(format: "%.2f", config.loraDropout))
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $config.loraDropout, in: 0...0.5, step: 0.01)
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Datasets Tab

struct DatasetsTab: View {
    var body: some View {
        VStack {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("No Datasets")
                .font(.headline)
            
            Text("Import training data in JSONL or CSV format")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Button("Import Dataset...") {
                // Show file picker
            }
            .buttonStyle(.borderedProminent)
            .padding(.top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Supporting Views

struct StatusBadge: View {
    let text: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(text)
                .font(.caption.bold())
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.15))
        .foregroundColor(color)
        .cornerRadius(4)
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Spacer()
            }
            
            Text(value)
                .font(.title2.bold())
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(minWidth: 120)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
    }
}

struct AdapterCard: View {
    let label: String
    let path: String
    let isBest: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption.bold())
                if isBest {
                    Image(systemName: "star.fill")
                        .font(.caption2)
                        .foregroundColor(.yellow)
                }
            }
            
            Text(path)
                .font(.caption.monospaced())
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
    }
}

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

struct TrainView_Previews: PreviewProvider {
    static var previews: some View {
        TrainView()
            .environmentObject(AppState())
    }
}
