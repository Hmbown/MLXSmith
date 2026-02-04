import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Providers } from "@/components/layout/providers";
import { MobileNav } from "@/components/layout/mobile-nav";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "MLXSmith - MLX Model Management",
  description: "Web interface for MLXSmith - Train, serve, and chat with MLX models",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} app-shell`}>
        <Providers>
          <div className="flex h-screen overflow-hidden">
            {/* Sidebar */}
            <aside className="hidden w-64 flex-col md:flex">
              <Sidebar />
            </aside>

            {/* Main Content */}
            <main className="flex flex-1 flex-col overflow-hidden">
              <MobileNav />
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
