import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Nunito_Sans } from "next/font/google";

import { QueryProvider } from "@/lib/query";

import "./globals.css";

const nunito = Nunito_Sans({ variable: "--font-nunito", subsets: ["latin"], display: "swap" });
const jetbrains = JetBrains_Mono({ variable: "--font-jetbrains", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: { default: "TreadOps", template: "%s · TreadOps" },
  description: "YOLO tyre-casing triage with a governed MLOps release loop and a circularity copilot.",
};

export const viewport: Viewport = { themeColor: "#6f5ff0", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${nunito.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
