// ContentView.swift
// Main content view with navigation split view

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        NavigationSplitView {
            Sidebar(selectedTab: $appState.selectedTab)
                .frame(minWidth: 200)
        } detail: {
            detailView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
    
    @ViewBuilder
    private var detailView: some View {
        switch appState.selectedTab {
        case .discover:
            DiscoverView()
        case .downloaded:
            DownloadedView()
        case .chat:
            ChatView()
        case .train:
            TrainView()
        case .serve:
            ServeView()
        case .settings:
            SettingsView()
        }
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @Binding var selectedTab: SidebarTab
    
    var body: some View {
        List(SidebarTab.allCases, selection: $selectedTab) { tab in
            NavigationLink(value: tab) {
                Label(tab.label, systemImage: tab.icon)
            }
        }
        .navigationTitle("MLXSmith")
        .listStyle(.sidebar)
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(AppState())
    }
}
