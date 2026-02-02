// MLXSmithAPI.swift
// API client for MLXSmith server

import Foundation

@MainActor
class MLXSmithAPI: ObservableObject {
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var isServerReachable = false
    
    private var baseURL: URL {
        UserDefaults.standard.string(forKey: "serverHost").flatMap { host in
            let port = UserDefaults.standard.integer(forKey: "serverPort")
            return URL(string: "http://\(host):\(port == 0 ? 8080 : port)")
        } ?? URL(string: "http://localhost:8080")!
    }
    
    private let urlSession: URLSession
    
    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 300
        self.urlSession = URLSession(configuration: config)
    }
    
    // MARK: - Health Check
    
    func checkHealth() async -> Bool {
        do {
            let (_, response) = try await fetch("/health")
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }
    
    // MARK: - Models
    
    func listLocalModels() async throws -> [LocalModel] {
        let (data, _) = try await fetch("/internal/models/list")
        
        // Parse the response - mock structure for now
        // In real implementation, server should return proper JSON
        if let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return json.compactMap { dict in
                guard let id = dict["id"] as? String,
                      let path = dict["path"] as? String,
                      let name = dict["name"] as? String else { return nil }
                
                return LocalModel(
                    id: id,
                    path: path,
                    name: name,
                    size: dict["size"] as? Int64 ?? 0,
                    quantization: dict["quantization"] as? String ?? "none",
                    lastUsed: (dict["last_used"] as? String).flatMap { parseDate($0) },
                    metadata: nil
                )
            }
        }
        
        // Return mock data for now
        return mockLocalModels()
    }
    
    func pullModel(modelId: String, quantize: Bool = false, qBits: Int? = nil) async throws {
        let request = PullModelRequest(modelId: modelId, quantize: quantize, qBits: qBits)
        let body = try JSONEncoder().encode(request)
        _ = try await post("/internal/models/pull", body: body)
    }
    
    func deleteModel(modelId: String) async throws {
        _ = try await post("/internal/models/delete", body: nil, queryItems: [
            URLQueryItem(name: "model_id", value: modelId)
        ])
    }
    
    // MARK: - Chat
    
    func sendChatMessage(
        messages: [ChatMessage],
        model: String? = nil,
        maxTokens: Int = 256,
        temperature: Double = 0.7,
        topP: Double = 1.0,
        topK: Int? = nil,
        stream: Bool = true
    ) async throws -> AsyncThrowingStream<String, Error> {
        let apiMessages = messages.map { APIMessage(role: $0.role.rawValue, content: $0.content) }
        let request = ChatRequest(
            model: model,
            messages: apiMessages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            stream: stream,
            stop: nil
        )
        
        let body = try JSONEncoder().encode(request)
        
        if stream {
            return try await streamChatCompletion(body: body)
        } else {
            let (data, _) = try await post("/v1/chat/completions", body: body)
            let response = try JSONDecoder().decode(ChatResponse.self, from: data)
            return AsyncThrowingStream { continuation in
                if let content = response.choices.first?.message?.content {
                    continuation.yield(content)
                }
                continuation.finish()
            }
        }
    }
    
    private func streamChatCompletion(body: Data) async throws -> AsyncThrowingStream<String, Error> {
        guard let url = URL(string: baseURL.absoluteString + "/v1/chat/completions") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let (bytes, response) = try await urlSession.bytes(for: request)
                    
                    guard let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 else {
                        continuation.finish(throwing: APIError.serverError("Invalid response"))
                        return
                    }
                    
                    var buffer = ""
                    for try await byte in bytes {
                        buffer.append(Character(UnicodeScalar(byte)))
                        
                        if buffer.hasSuffix("\n\n") {
                            let lines = buffer.split(separator: "\n")
                            for line in lines {
                                let trimmed = line.trimmingCharacters(in: .whitespaces)
                                if trimmed.hasPrefix("data: ") {
                                    let data = String(trimmed.dropFirst(6))
                                    if data == "[DONE]" {
                                        continuation.finish()
                                        return
                                    }
                                    if let jsonData = data.data(using: .utf8),
                                       let chunk = try? JSONDecoder().decode(ChatResponse.self, from: jsonData),
                                       let content = chunk.choices.first?.delta?.content {
                                        continuation.yield(content)
                                    }
                                }
                            }
                            buffer = ""
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - RLM
    
    func getRLMState() async throws -> RLMState {
        let (data, _) = try await fetch("/internal/rlm/state")
        return try JSONDecoder().decode(RLMState.self, from: data)
    }
    
    func getRLMHistory() async throws -> [RLMHistoryEntry] {
        let (data, _) = try await fetch("/internal/rlm/history")
        return try JSONDecoder().decode([RLMHistoryEntry].self, from: data)
    }
    
    // MARK: - HF Token
    
    func setHFToken(_ token: String) async throws {
        let body = try JSONSerialization.data(withJSONObject: ["token": token])
        _ = try await post("/internal/hf/token", body: body)
    }
    
    // MARK: - Private Helpers
    
    private func fetch(_ path: String) async throws -> (Data, URLResponse) {
        guard let url = URL(string: baseURL.absoluteString + path) else {
            throw APIError.invalidURL
        }
        return try await urlSession.data(from: url)
    }
    
    private func post(_ path: String, body: Data?, queryItems: [URLQueryItem]? = nil) async throws -> (Data, URLResponse) {
        var components = URLComponents(string: baseURL.absoluteString + path)
        components?.queryItems = queryItems
        
        guard let url = components?.url else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        
        return try await urlSession.data(for: request)
    }
    
    private func parseDate(_ string: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        return formatter.date(from: string)
    }
    
    // MARK: - Mock Data
    
    private func mockLocalModels() -> [LocalModel] {
        [
            LocalModel(
                id: "llama-3.2-3b-4bit",
                path: "~/.mlxsmith/cache/mlx/Llama-3.2-3B-Instruct-4bit",
                name: "Llama 3.2 3B (4-bit)",
                size: 1_800_000_000,
                quantization: "4-bit",
                lastUsed: Date(),
                metadata: ModelMetadata(
                    baseModel: "meta-llama/Llama-3.2-3B-Instruct",
                    quantization: "4-bit",
                    architecture: "Llama"
                )
            ),
            LocalModel(
                id: "qwen2.5-7b-4bit",
                path: "~/.mlxsmith/cache/mlx/Qwen2.5-7B-Instruct-4bit",
                name: "Qwen 2.5 7B (4-bit)",
                size: 4_200_000_000,
                quantization: "4-bit",
                lastUsed: nil,
                metadata: nil
            ),
            LocalModel(
                id: "adapter-coding-v1",
                path: "runs/sft_001/adapter",
                name: "Coding Adapter v1",
                size: 150_000_000,
                quantization: "none",
                lastUsed: Date().addingTimeInterval(-86400),
                metadata: ModelMetadata(
                    baseModel: "Llama-3.2-3B",
                    quantization: nil,
                    architecture: "LoRA"
                )
            )
        ]
    }
}

// MARK: - Errors

enum APIError: Error, LocalizedError {
    case invalidURL
    case serverError(String)
    case decodingError
    case networkError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .serverError(let message):
            return "Server error: \(message)"
        case .decodingError:
            return "Failed to decode response"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

// MARK: - HuggingFace API

@MainActor
class HuggingFaceAPI: ObservableObject {
    @Published var models: [HFModel] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let baseURL = URL(string: "https://huggingface.co/api")!
    private let urlSession = URLSession.shared
    
    func searchModels(query: String, limit: Int = 20) async {
        isLoading = true
        defer { isLoading = false }
        
        // In production, this would call the actual HF API
        // For now, return mock data
        models = mockHFModels()
    }
    
    func fetchTrendingModels(limit: Int = 20) async {
        isLoading = true
        defer { isLoading = false }
        
        models = mockHFModels()
    }
    
    private func mockHFModels() -> [HFModel] {
        [
            HFModel(
                id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                author: "mlx-community",
                description: "Meta's Llama 3.2 3B instruction-tuned model, quantized to 4-bit for MLX",
                tags: ["mlx", "llama", "text-generation", "4-bit"],
                downloads: 125000,
                likes: 850,
                lastModified: Date(),
                pipelineTag: "text-generation",
                siblings: nil
            ),
            HFModel(
                id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                author: "mlx-community",
                description: "Qwen 2.5 7B instruction model, MLX-compatible 4-bit quantization",
                tags: ["mlx", "qwen", "text-generation", "multilingual"],
                downloads: 89000,
                likes: 620,
                lastModified: Date(),
                pipelineTag: "text-generation",
                siblings: nil
            ),
            HFModel(
                id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                modelId: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                author: "mlx-community",
                description: "Mistral 7B Instruct v0.3, 4-bit quantized for Apple Silicon",
                tags: ["mlx", "mistral", "text-generation", "4-bit"],
                downloads: 156000,
                likes: 1100,
                lastModified: Date(),
                pipelineTag: "text-generation",
                siblings: nil
            ),
            HFModel(
                id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
                modelId: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
                author: "mlx-community",
                description: "DeepSeek R1 Distilled on Qwen 7B, reasoning model for MLX",
                tags: ["mlx", "deepseek", "qwen", "reasoning", "4-bit"],
                downloads: 45000,
                likes: 420,
                lastModified: Date(),
                pipelineTag: "text-generation",
                siblings: nil
            ),
            HFModel(
                id: "mlx-community/Phi-4-mini-instruct-4bit",
                modelId: "mlx-community/Phi-4-mini-instruct-4bit",
                author: "mlx-community",
                description: "Microsoft Phi-4 mini instruction-tuned, compact and efficient",
                tags: ["mlx", "phi", "text-generation", "4-bit"],
                downloads: 32000,
                likes: 280,
                lastModified: Date(),
                pipelineTag: "text-generation",
                siblings: nil
            )
        ]
    }
}
