# Prime Intellect Dashboard - Design Notes for MLXSmith SwiftUI App

*Compiled 2026-02-01 from analysis of app.primeintellect.ai*

---

## 1. Overall Design Philosophy

Prime Intellect's dashboard embodies **functional minimalism** - every element earns its place. The design prioritizes:

- **Dark mode as default** (forced `dark` class on HTML root)
- **High contrast**: Black/very dark backgrounds (#0E0E0E) with white/light text
- **Information density without clutter**: Dense data, generous spacing
- **Action-oriented layout**: Every screen has a clear primary CTA
- **Progressive disclosure**: Advanced options collapsed by default

### Key Takeaway for MLXSmith
The simplicity comes from *restraint*, not from lack of features. Prime Intellect has 40+ GPU types, complex cluster configs, environment hubs - but presents them through **clean grids, clear hierarchy, and one-thing-at-a-time flows**.

---

## 2. Navigation Architecture

### Sidebar Structure (Grouped by Function)
```
COMPUTE
  Single-Node GPUs
  Multi-Node Cluster
  Instances
  Templates
  Reserved Instances

LAB
  Environments Hub
  Evaluations
  Training

ACCOUNT
  Profile
  Inbox
  Billing
  API Keys

SUPPORT
  Chat
  Documentation (external link)
```

### Design Details
- **Section headers** are uppercase category labels (COMPUTE, LAB, ACCOUNT)
- **Each item** has an SVG icon + text label
- **Active state** is highlighted (likely with accent color or background)
- **Collapsible sidebar** via "Toggle Sidebar" control
- **Fixed position** - sidebar doesn't scroll with content
- Simple, flat hierarchy - no nested sub-menus

### MLXSmith Mapping
```
MODELS
  Discover        (HF browser)
  Downloaded      (local cache)

TRAINING
  Fine-Tune       (SFT/Pref/RFT)
  RLM Loop        (autonomous training)
  Runs            (history + artifacts)

INFERENCE
  Chat            (conversation UI)
  Serve           (API server)

SYSTEM
  Dashboard       (metrics/monitoring)
  Settings        (config)
```

**SwiftUI implementation**: `NavigationSplitView` with `List` sidebar, grouped by `Section` headers using SF Symbols for icons.

---

## 3. Dashboard Home (Action Cards Grid)

The home page uses a **3-column grid of action cards** organized by category:

### Card Anatomy
```
┌──────────────────────────────┐
│  [Icon]                       │
│                               │
│  Card Title                   │
│  Brief description of what    │
│  this action does             │
│                               │
│  [Implicit link - entire      │
│   card is tappable]           │
└──────────────────────────────┘
```

### Card Categories on Home
**Compute** (3 cards):
- Deploy GPU Instance
- Deploy Multi-Node Cluster
- Request Reserved Instances

**Lab** (3 cards):
- Environments Hub
- Run Evaluations
- Run RL Training

**Account** (3 cards):
- Manage Billing
- Create API Key
- Edit Profile

### Design Notes
- Cards are **link-based** - entire card is clickable, navigates to relevant page
- **No buttons inside cards** - the card itself IS the action
- Minimal text: title + one line description
- Even spacing, consistent card sizes
- Category headers above each row of cards

### MLXSmith Dashboard Home Equivalent
```
QUICK ACTIONS (3-column grid)

┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Pull Model  │ │  Start SFT  │ │  Open Chat  │
│  Download    │ │  Fine-tune   │ │  Talk to     │
│  from HF     │ │  a model     │ │  your model  │
└─────────────┘ └─────────────┘ └─────────────┘

┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Start RLM  │ │  Launch API │ │  View Runs  │
│  Autonomous  │ │  OpenAI-     │ │  Training    │
│  training    │ │  compatible  │ │  history     │
└─────────────┘ └─────────────┘ └─────────────┘
```

**SwiftUI**: `LazyVGrid(columns: [GridItem(.flexible()), ...], spacing: 16)` with `NavigationLink` wrapping each card.

---

## 4. GPU/Resource Selection (Create Cluster Page)

This is the richest UI page - a **multi-step configuration wizard**:

### Step 1: GPU Grid Selection
- **40+ GPU options** displayed in a scrollable grid
- Each GPU shown as a **selectable card** with:
  - NVIDIA logo or CPU icon
  - Model name (e.g., "H100", "RTX 4090")
  - VRAM capacity (e.g., "80GB")
  - Socket type (e.g., "SXM5", "PCIe")
- **Selection state**: Checkmark/circle indicator on selected card
- **Filter bar** above grid:
  - Location dropdown ("Any")
  - "Show Only Available" toggle
  - Search/filter text input

### Step 2: Base Image Selection
- **6-8 pre-configured Docker images** as radio-button cards
- Each shows: name, CUDA version, description
- "Create Custom Template" option at the end
- Descriptions are concise (one sentence explaining use case)

### Step 3: Advanced Options (Collapsed)
- CPU range slider
- RAM range slider
- Deployment type dropdown
- Disk selection
- Reset button

### Step 4: Summary + Deploy
- Review selected configuration
- Primary "Continue" CTA button

### MLXSmith Equivalent: Model Selection
```
MODEL BROWSER
┌──────────────────────────────────────────────┐
│  Filter: [Architecture v] [Size v] [Quant v] │
│  Search: [________________________] [Search]  │
├──────────────────────────────────────────────┤
│                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ Qwen3   │ │ Llama4  │ │ Gemma3  │        │
│  │ 4B      │ │ 8B      │ │ 2B      │        │
│  │ Q4      │ │ Q6      │ │ FP16    │        │
│  │ 2.1 GB  │ │ 5.4 GB  │ │ 4.0 GB  │        │
│  │ [Pull]  │ │ [Pull]  │ │ [Local] │        │
│  └─────────┘ └─────────┘ └─────────┘        │
│                                               │
└──────────────────────────────────────────────┘
```

**Key pattern**: Grid of selectable cards + filter bar above. The grid IS the selection mechanism.

---

## 5. Environments Hub

### Layout
- **Header**: "Environments Hub" + description text
- **Two action buttons**: "Learn More" + "Create Environment"
- **Tab navigation**: "Explore" | "My Stars"
- **Community metric**: "100+" badge
- **Grid of environment cards** (content varies)

### Design Pattern
This follows the **Browse + Create** pattern:
1. Descriptive header explaining the page
2. Primary CTA to create new content
3. Tabs to filter existing content
4. Grid/list of community content cards

### MLXSmith Equivalent: Environments / Training Configs
```
ENVIRONMENTS
─────────────────────────────────
Discover and manage RL training environments

[Browse Community]  [Create New]

[My Envs] | [Community] | [Starred]

┌────────────────┐ ┌────────────────┐
│  Code Verifier │ │  Math Proofs   │
│  Python pytest │ │  Lean4 checker │
│  12 tasks      │ │  45 tasks      │
│  ★ 24          │ │  ★ 67          │
└────────────────┘ └────────────────┘
```

---

## 6. Color System & Typography

### Colors (Dark Theme)
| Element | Color |
|---------|-------|
| Background (primary) | `#0E0E0E` (near-black) |
| Background (sidebar) | Slightly lighter dark |
| Background (cards) | `#1A1A1A` - `#222` range |
| Text (primary) | White `#FFFFFF` |
| Text (secondary) | Light gray `#A0A0A0` |
| Text (muted) | `#666666` |
| Accent/Interactive | Blue or brand color |
| Borders | Subtle `#333` or transparent |
| Hover states | Border animation / glow |

### Typography
| Use | Font | Weight |
|-----|------|--------|
| Body text | IBM Plex Sans | Regular (400) |
| Code/model names | IBM Plex Mono / Fira Mono | Regular |
| Headings | IBM Plex Sans | Bold (700) |
| Labels/captions | Inter | Medium (500) |
| Navigation | Sans-serif system | Medium |

### MLXSmith Typography (Native)
| Use | Font |
|-----|------|
| Body | SF Pro (system default) |
| Code/model names | SF Mono / `.monospaced` |
| Headings | SF Pro Display / `.title` |
| Labels | SF Pro Text / `.caption` |
| Numbers/metrics | SF Mono / `.monospacedDigit` |

Use SwiftUI's built-in text styles (`.title`, `.headline`, `.body`, `.caption`) for automatic dynamic type support.

---

## 7. Interaction Patterns

### Pattern: Card Grid Selection
**Used for**: GPU selection, template selection, model browsing
- Grid of cards, each representing one option
- Single-select: tap to select, checkmark appears
- Filter/search bar above
- Cards show 3-4 key attributes max

### Pattern: Tab Navigation
**Used for**: Environments Hub (Explore/My Stars), sub-pages
- Horizontal tab bar below page header
- Active tab has underline/highlight indicator
- Content below changes per tab

### Pattern: Progressive Disclosure
**Used for**: Advanced cluster options, detailed settings
- Default: collapsed/hidden
- "Advanced options" toggle to expand
- Reset button to clear advanced filters
- Keeps primary flow simple, power users can dig in

### Pattern: Action Header
**Used for**: Every page
- Page title (large heading)
- Description text (one line, muted)
- Primary CTA button(s) aligned right
- Example: "Environments Hub" + "Create Environment" button

### Pattern: Real-time Feedback
**Used for**: Pricing, deployment status
- "Prices update in realtime" - live data without page refresh
- Status indicators for running instances
- Feedback widget ("How can we improve?") on every page

### Pattern: Wizard/Flow
**Used for**: Cluster creation
- Multi-step form with clear progression
- Each step focuses on one decision
- Summary/review before final action
- Primary CTA to advance ("Continue")

---

## 8. Landing Page Design Language

### Hero Section
- **Bold, centered messaging**: "The Open Superintelligence Stack"
- Large background image with gradient overlay
- Strong call-to-action: "Get started"
- Social proof below: backed-by logos

### Feature Cards
- Hover animations (border glow/animation)
- 3-column grid for service tiers
- Minimal text per card

### Navigation
- Fixed header: Logo | Compute | LAB | Research | Docs | Blog | Login
- Clear CTA button stands out from nav links

### MLXSmith Onboarding / Welcome Screen
```
┌──────────────────────────────────────────┐
│                                          │
│          MLXSmith                         │
│    Train AI models on your Mac            │
│                                          │
│    [Get Started]                          │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Pull     │ │ Train    │ │ Chat     │ │
│  │ Models   │ │ Locally  │ │ & Serve  │ │
│  │ from HF  │ │ with MLX │ │ APIs     │ │
│  └──────────┘ └──────────┘ └──────────┘ │
│                                          │
└──────────────────────────────────────────┘
```

---

## 9. Key Simplicity Principles Extracted

### 1. One Primary Action Per Screen
Every page has ONE obvious thing to do. The cluster page's CTA is "Continue". The environments page has "Create Environment". Don't overwhelm with choices.

### 2. Grid > List for Browsable Content
GPUs, templates, environments - all use card grids. Grids allow quick visual scanning. Lists are for history/logs.

### 3. Category Headers Create Order
Sidebar: COMPUTE / LAB / ACCOUNT. Dashboard: Compute Cards / Lab Cards / Account Cards. Group related items, label the groups.

### 4. Filter First, Then Browse
GPU selection has filters above the grid. Same pattern works for model browsing. Let users narrow before they scan.

### 5. Dark Mode Creates Focus
Dark backgrounds make content pop. Cards and interactive elements stand out against the dark canvas. Less visual noise.

### 6. Descriptions Are One Line
"Deploy GPU Instance" -> "single-node on-demand option". Keep it short. If it needs more, it goes on the detail page.

### 7. Feedback Is Always Available
Every page has "How can we improve?" Accessible but not intrusive.

### 8. Advanced = Hidden by Default
Power features exist but don't pollute the primary flow. Collapsed sections, expandable panels.

---

## 10. Direct MLXSmith SwiftUI Implementation Guide

### Minimum Viable Design System

```swift
// Colors
extension Color {
    static let appBackground = Color(hex: "#0E0E0E")
    static let cardBackground = Color(hex: "#1A1A1A")
    static let cardBorder = Color(hex: "#333333")
    static let textPrimary = Color.white
    static let textSecondary = Color(hex: "#A0A0A0")
    static let textMuted = Color(hex: "#666666")
    static let accentBrand = Color.blue  // or custom brand color
}

// Card Component
struct ActionCard: View {
    let icon: String      // SF Symbol name
    let title: String
    let description: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(.accentBrand)
            Text(title)
                .font(.headline)
                .foregroundStyle(.textPrimary)
            Text(description)
                .font(.caption)
                .foregroundStyle(.textSecondary)
                .lineLimit(2)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(.cardBorder, lineWidth: 1)
        )
    }
}

// Sidebar Section (Prime Intellect-style grouping)
struct SidebarSection: View {
    let title: String
    let items: [(icon: String, label: String, destination: AppTab)]

    var body: some View {
        Section(title) {
            ForEach(items, id: \.label) { item in
                NavigationLink(value: item.destination) {
                    Label(item.label, systemImage: item.icon)
                }
            }
        }
    }
}

// Page Header (every page gets this)
struct PageHeader: View {
    let title: String
    let description: String
    var action: (() -> Void)? = nil
    var actionLabel: String = ""

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.largeTitle).bold()
                Text(description).font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer()
            if let action {
                Button(actionLabel, action: action)
                    .buttonStyle(.borderedProminent)
            }
        }
    }
}
```

### Navigation Structure

```swift
struct ContentView: View {
    @State private var selection: AppTab? = .home

    var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                Section("MODELS") {
                    Label("Discover", systemImage: "sparkle.magnifyingglass")
                        .tag(AppTab.discover)
                    Label("Downloaded", systemImage: "arrow.down.circle.fill")
                        .tag(AppTab.downloaded)
                }
                Section("TRAINING") {
                    Label("Fine-Tune", systemImage: "brain")
                        .tag(AppTab.train)
                    Label("RLM Loop", systemImage: "arrow.trianglehead.2.clockwise")
                        .tag(AppTab.rlm)
                    Label("Runs", systemImage: "clock.arrow.circlepath")
                        .tag(AppTab.runs)
                }
                Section("INFERENCE") {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .tag(AppTab.chat)
                    Label("Serve", systemImage: "server.rack")
                        .tag(AppTab.serve)
                }
                Section("SYSTEM") {
                    Label("Dashboard", systemImage: "chart.bar")
                        .tag(AppTab.dashboard)
                    Label("Settings", systemImage: "gearshape")
                        .tag(AppTab.settings)
                }
            }
            .navigationTitle("MLXSmith")
        } detail: {
            switch selection {
            case .home: HomeView()
            case .discover: DiscoverView()
            // ...
            default: HomeView()
            }
        }
    }
}
```

---

## 11. What NOT to Copy

1. **Next.js/React patterns** - Don't replicate web conventions in SwiftUI. Use native patterns (NavigationSplitView, not a sidebar div).
2. **The feedback widget** - Not needed in a native app (use App Store reviews / GitHub issues).
3. **The sign-in redirect page** - SwiftUI handles auth differently (ASWebAuthenticationSession).
4. **GTM/Analytics scripts** - Use native TelemetryDeck or similar if needed.
5. **Server-side rendering tricks** - SwiftUI is declarative and client-side. Embrace it.

---

## Summary: The Prime Intellect Formula

```
Dark theme
    + Sidebar navigation (grouped by function)
    + Card grids for browsable content
    + One primary action per page
    + Progressive disclosure for advanced options
    + Minimal text (title + one-line description)
    + Filter bar above content grids
    + Real-time status/metrics where relevant
    = Clean, professional, approachable dashboard
```

Apply this formula to MLXSmith's domain (models, training, inference) and you get a native macOS app that feels as polished as Prime Intellect's web dashboard, but with the performance and integration advantages of SwiftUI on Apple Silicon.
