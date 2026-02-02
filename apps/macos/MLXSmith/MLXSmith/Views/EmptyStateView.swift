// EmptyStateView.swift
// Reusable empty state component

import SwiftUI

struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?
    
    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            
            Image(systemName: icon)
                .font(.system(size: 56))
                .foregroundColor(.secondary.opacity(0.6))
            
            Text(title)
                .font(.title2.bold())
            
            Text(message)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
            
            if let actionTitle = actionTitle, let action = action {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)
            }
            
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Preview

struct EmptyStateView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            EmptyStateView(
                icon: "square.stack.3d.up.slash",
                title: "No Models",
                message: "Get started by downloading a model from HuggingFace"
            )
            
            EmptyStateView(
                icon: "bubble.left.and.exclamationmark.bubble.right",
                title: "No Messages",
                message: "Start a conversation by typing a message below",
                actionTitle: "Start Chat",
                action: {}
            )
        }
        .padding()
    }
}
