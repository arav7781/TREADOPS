import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Noto_Sans, Saira } from "next/font/google";

import { QueryProvider } from "@/lib/query";

import "./globals.css";

// Michelin's site sets body copy in Noto Sans. Its headline face (wide, heavy italic) is proprietary; Saira at 125%
// width is the closest open-source match, so headlines use it with the width axis.
const noto = Noto_Sans({ variable: "--font-noto", subsets: ["latin"], display: "swap" });
const saira = Saira({ variable: "--font-saira", subsets: ["latin"], style: ["normal", "italic"], axes: ["wdth"], display: "swap" });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: { default: "TreadOps", template: "%s · TreadOps" },
  description: "Tyre-casing triage with a governed MLOps release loop and a circularity copilot.",
};

export const viewport: Viewport = { themeColor: "#000000", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${noto.variable} ${saira.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
