import type { Metadata, Viewport } from "next";
import { Barlow, Barlow_Semi_Condensed, JetBrains_Mono } from "next/font/google";

import { QueryProvider } from "@/lib/query";

import "./globals.css";

// Barlow: a free, sturdy grotesque in the spirit of Michelin's (proprietary) brand type.
const barlow = Barlow({ variable: "--font-barlow", subsets: ["latin"], weight: ["400", "500", "600", "700", "800"], display: "swap" });
const barlowCondensed = Barlow_Semi_Condensed({ variable: "--font-barlow-condensed", subsets: ["latin"], weight: ["600", "700", "800"], display: "swap" });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: { default: "TreadOps", template: "%s · TreadOps" },
  description: "YOLO tyre-casing triage with a governed MLOps release loop and a circularity copilot.",
};

export const viewport: Viewport = { themeColor: "#00205b", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${barlow.variable} ${barlowCondensed.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
