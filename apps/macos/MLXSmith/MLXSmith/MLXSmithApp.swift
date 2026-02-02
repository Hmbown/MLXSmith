// MLXSmithApp.swift
// MLXSmith macOS App

import SwiftUI

@main
struct MLXSmithApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 1000, minHeight: 700)
        }
        .windowStyle(.titleBar)
        .defaultPosition(.center)
        
        Settings {
            SettingsView()
                .environmentObject(appState)
                .frame(width: 500, height: 400)
        }
    }
}

// MARK: - App State

@MainActor
class AppState: ObservableObject {
    @Published var selectedTab: SidebarTab = .discover
    @Published var apiClient = MLXSmithAPI()
    @Published var keychain = KeychainManager()
    @Published var serverProcess: ServerProcessManager?
    
    // Project configuration
    @Published var projectRoot: String {
        didSet {
            UserDefaults.standard.set(projectRoot, forKey: "projectRoot")
        }
    }
    @Published var cachePath: String {
        didSet {
            UserDefaults.standard.set(cachePath, forKey: "cachePath")
        }
    }
    @Published var serverPort: Int {
        didSet {
            UserDefaults.standard.set(serverPort, forKey: "serverPort")
        }
    }
    @Published var serverHost: String {
        didSet {
            UserDefaults.standard.set(serverHost, forKey: "serverHost")
        }
    }
    
    init() {
        self.projectRoot = UserDefaults.standard.string(forKey: "projectRoot") 
            ?? FileManager.default.homeDirectoryForCurrentUser.path
        self.cachePath = UserDefaults.standard.string(forKey: "cachePath")
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".mlxsmith/cache").path
        self.serverPort = UserDefaults.standard.integer(forKey: "serverPort") != 0 
            ? UserDefaults.standard.integer(forKey: "serverPort") : 8080
        self.serverHost = UserDefaults.standard.string(forKey: "serverHost") ?? "localhost"
    }
}

// MARK: - Sidebar Tabs

enum SidebarTab: String, CaseIterable, Identifiable {
    case discover = "Discover"
    case downloaded = "Downloaded"
    case chat = "Chat"
    case train = "Train"
    case serve = "Serve"
    case settings = "Settings"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .discover: return "magnifyingglass"
        case .downloaded: return "square.grid.2x2"
        case .chat: return "bubble.left.and.bubble.right"
        case .train: return "figure.strengthtraining.traditional"
        case .serve: return "server.rack"
        case .settings: return "gearshape"
        }
    }
    
    var label: String {
        switch self {
        case .discover: return "Discover"
        case .downloaded: return "Downloaded"
        case .chat: return "Chat"
        case .train: return "Train"
        case .serve: return "Serve"
        case .settings: return "Settings"
        }
    }
}
