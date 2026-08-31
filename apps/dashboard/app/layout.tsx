import type { Metadata } from "next";
import "./globals.css";
import { NavBar } from "@/components/NavBar";

export const metadata: Metadata = {
  title: "Genesis Kit — Autonomous Incident Commander",
  description: "Reasoning trail, causal graph, and cost dashboard for the incident-response agent.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-[var(--background)] font-sans text-[var(--foreground)]">
        <NavBar />
        {children}
      </body>
    </html>
  );
}
