// KeychainManager.swift
// Keychain access for secure token storage

import Foundation
import Security

@MainActor
class KeychainManager: ObservableObject {
    @Published var hfToken: String? {
        didSet {
            if let token = hfToken {
                _ = saveToken(token, for: "huggingface_token")
            } else {
                _ = deleteToken(for: "huggingface_token")
            }
        }
    }
    
    private let service = "com.mlxsmith.app"
    
    init() {
        self.hfToken = loadToken(for: "huggingface_token")
    }
    
    // MARK: - Token Management
    
    func saveToken(_ token: String, for account: String) -> Bool {
        let data = token.data(using: .utf8)!
        
        // Delete existing item first
        deleteToken(for: account)
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    func loadToken(for account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let data = result as? Data,
              let token = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        return token
    }
    
    func deleteToken(for account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
    
    // MARK: - API Key Management
    
    func saveAPIKey(_ key: String, provider: String) -> Bool {
        return saveToken(key, for: "api_key_\(provider)")
    }
    
    func loadAPIKey(for provider: String) -> String? {
        return loadToken(for: "api_key_\(provider)")
    }
    
    func deleteAPIKey(for provider: String) -> Bool {
        return deleteToken(for: "api_key_\(provider)")
    }
}

// MARK: - Server Process Manager

@MainActor
class ServerProcessManager: ObservableObject {
    @Published var isRunning = false
    @Published var logs: [ServerLog] = []
    @Published var currentPort: Int = 8080
    @Published var currentModel: String?
    
    private var process: Process?
    private var pipe: Pipe?
    
    func startServer(model: String, port: Int, projectRoot: String) async throws {
        guard !isRunning else { return }
        
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = [
            "python", "-m", "mlxsmith",
            "serve",
            "--model", model,
            "--port", String(port)
        ]
        task.currentDirectoryURL = URL(fileURLWithPath: projectRoot)
        
        let outputPipe = Pipe()
        task.standardOutput = outputPipe
        task.standardError = outputPipe
        
        let handler = outputPipe.fileHandleForReading
        handler.readabilityHandler = { [weak self] handle in
            guard let data = try? handle.read(upToCount: 1024),
                  let output = String(data: data, encoding: .utf8),
                  !output.isEmpty else { return }
            
            Task { @MainActor in
                self?.addLog(message: output, level: .info)
            }
        }
        
        try task.run()
        self.process = task
        self.pipe = outputPipe
        self.isRunning = true
        self.currentPort = port
        self.currentModel = model
        
        addLog(message: "Server starting on port \(port)...", level: .info)
    }
    
    func stopServer() {
        process?.terminate()
        process = nil
        pipe = nil
        isRunning = false
        addLog(message: "Server stopped", level: .info)
    }
    
    private func addLog(message: String, level: ServerLog.LogLevel, endpoint: String? = nil) {
        let lines = message.components(separatedBy: .newlines)
        for line in lines where !line.isEmpty {
            let log = ServerLog(
                timestamp: Date(),
                level: level,
                message: line,
                endpoint: endpoint
            )
            logs.append(log)
            
            // Keep only last 1000 logs
            if logs.count > 1000 {
                logs.removeFirst(logs.count - 1000)
            }
        }
    }
    
    deinit {
        stopServer()
    }
}
