// Formatters.swift
// Utility formatters for the app

import Foundation

struct Formatters {
    static let numberFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 2
        return formatter
    }()
    
    static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter
    }()
    
    static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()
    
    static let relativeDateFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter
    }()
    
    static func formatTokensPerSecond(_ tps: Double) -> String {
        if tps >= 1000 {
            return String(format: "%.1fK tok/s", tps / 1000)
        }
        return String(format: "%.1f tok/s", tps)
    }
    
    static func formatNumber(_ number: Int) -> String {
        if number >= 1_000_000 {
            return String(format: "%.1fM", Double(number) / 1_000_000)
        } else if number >= 1_000 {
            return String(format: "%.1fK", Double(number) / 1_000)
        }
        return String(number)
    }
    
    static func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        } else if seconds < 3600 {
            let minutes = Int(seconds / 60)
            let secs = Int(seconds) % 60
            return String(format: "%dm %ds", minutes, secs)
        } else {
            let hours = Int(seconds / 3600)
            let minutes = Int(seconds / 60) % 60
            return String(format: "%dh %dm", hours, minutes)
        }
    }
}
