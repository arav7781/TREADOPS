import { describe, expect, it } from "vitest";

import { type FrameScore, selectFrames } from "@/lib/video";

const thumb = (v: number) => new Float32Array(256).fill(v);
const frame = (time: number, sharpness: number, brightness = 0.5, t = time / 10): FrameScore =>
  ({ time, sharpness, brightness, thumb: thumb(t) });

describe("selectFrames", () => {
  it("keeps the sharpest usable frame in each segment of the clip", () => {
    const cands = Array.from({ length: 12 }, (_, i) => frame(i, i % 2 ? 200 : 80));
    const picked = selectFrames(cands, 6);
    expect(picked.map((f) => f.time)).toEqual([1, 3, 5, 7, 9, 11]);
  });

  it("drops dark, overexposed and much-blurrier-than-usual frames", () => {
    const cands = [frame(0, 200, 0.02), frame(1, 200, 0.99), frame(2, 10), frame(3, 200), frame(4, 210), frame(5, 190)];
    const picked = selectFrames(cands, 3);
    expect(picked.every((f) => f.brightness > 0.08 && f.brightness < 0.97 && f.sharpness >= 100)).toBe(true);
    expect(picked.map((f) => f.time)).not.toContain(2);
  });

  it("drops near-duplicate neighbours when the tyre did not move", () => {
    const still = Array.from({ length: 12 }, (_, i) => frame(i, 100 + i, 0.5, 0.3));
    expect(selectFrames(still, 6)).toHaveLength(1);
  });

  it("keeps every sector of a slowly turning, uniform tread", () => {
    // Small but steady frame-to-frame change: a fixed threshold would merge these into one frame.
    const turning = Array.from({ length: 24 }, (_, i) => frame(i * 0.5, 150, 0.5, 0.3 + i * 0.006));
    expect(selectFrames(turning, 6)).toHaveLength(6);
  });

  it("returns nothing for an empty or unusable clip", () => {
    expect(selectFrames([], 6)).toEqual([]);
    expect(selectFrames([frame(0, 100, 0.01), frame(1, 100, 0.01)], 6)).toEqual([]);
  });
});
