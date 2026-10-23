import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  devIndicators: false,
  images: { qualities: [75] }, // Next 16 requires an explicit allowlist
  async redirects() {
    return [
      { source: "/", destination: "/dashboard", permanent: false },
      // The copilot and the inspector upload are one chat now; ?session= is passed through.
      { source: "/copilot", destination: "/inspect", permanent: false },
    ];
  },
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
