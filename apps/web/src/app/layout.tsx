import type { Metadata, Viewport } from "next";
import { Exo_2, JetBrains_Mono, Noto_Sans } from "next/font/google";

import { QueryProvider } from "@/lib/query";

import "./globals.css";

// Michelin's site sets body copy in Noto Sans and headlines in a heavy italic display face; Exo 2 is the closest
// open-source match for the latter (their own typeface is proprietary and not used here).
const noto = Noto_Sans({ variable: "--font-noto", subsets: ["latin"], display: "swap" });
const exo = Exo_2({ variable: "--font-exo", subsets: ["latin"], weight: ["300", "500", "700", "800", "900"], style: ["normal", "italic"], display: "swap" });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: { default: "TreadOps", template: "%s · TreadOps" },
  description: "YOLO tyre-casing triage with a governed MLOps release loop and a circularity copilot.",
};

export const viewport: Viewport = { themeColor: "#000000", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${noto.variable} ${exo.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
