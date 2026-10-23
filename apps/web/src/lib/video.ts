/**
 * Video input for Bib, done in the browser: sample a clip (or a live camera) at a few frames per second, score
 * each frame, and send only the few that are worth classifying. Nothing but the chosen JPEG frames is uploaded.
 *
 * Selection: frames that are too dark/bright or much blurrier than the clip's median are dropped; the clip is cut
 * into `k` equal segments (one per sector of the tyre's turn) and the sharpest frame of each is kept. A pick that
 * barely differs from the previous one, relative to how much this clip moves from frame to frame, is dropped (the
 * tyre did not turn); a fixed threshold would wrongly merge the sectors of a uniform tread. This is a pre-filter only: the worker's quality screen still
 * checks every frame it receives.
 */

export interface FrameScore {
  time: number;          // seconds into the clip / capture
  sharpness: number;     // variance of a 3x3 Laplacian on a <=320 px grey image
  brightness: number;    // mean grey level in [0, 1]
  thumb: Float32Array;   // 16x16 grey thumbnail, for duplicate detection
}

export interface SampledFrame extends FrameScore {
  blob: Blob;
  url: string;           // object URL for previews; revoke when done
}

export interface VideoSample {
  source: "clip" | "camera";
  duration: number;
  considered: number;
  frames: SampledFrame[];
}

export const MAX_FRAMES = 6;
export const VIDEO_TYPES = ["video/mp4", "video/webm", "video/quicktime"];
const MIN_BRIGHTNESS = 0.08;
const MAX_BRIGHTNESS = 0.97;
const STILL_DIFF = 0.004; // below this, two frames are the same view (sensor noise only)

/** Score one rendered frame. `canvas` holds the frame; a small working canvas is reused. */
export function scoreCanvas(canvas: HTMLCanvasElement, work: HTMLCanvasElement, time: number): FrameScore {
  const side = 320;
  const scale = Math.min(1, side / Math.max(canvas.width, canvas.height));
  const w = Math.max(3, Math.round(canvas.width * scale));
  const h = Math.max(3, Math.round(canvas.height * scale));
  work.width = w;
  work.height = h;
  const g = work.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(canvas, 0, 0, w, h);
  const { data } = g.getImageData(0, 0, w, h);
  const grey = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) {
    const y = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
    grey[i] = y;
    sum += y;
  }
  let lapSum = 0;
  let lapSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = (4 * grey[i] - grey[i - 1] - grey[i + 1] - grey[i - w] - grey[i + w]) * 255;
      lapSum += lap;
      lapSq += lap * lap;
      n += 1;
    }
  }
  const mean = lapSum / Math.max(n, 1);
  const thumb = new Float32Array(256);
  for (let ty = 0; ty < 16; ty++) {
    for (let tx = 0; tx < 16; tx++) {
      thumb[ty * 16 + tx] = grey[Math.min(h - 1, Math.floor(((ty + 0.5) * h) / 16)) * w + Math.min(w - 1, Math.floor(((tx + 0.5) * w) / 16))];
    }
  }
  return { time, sharpness: lapSq / Math.max(n, 1) - mean * mean, brightness: sum / (w * h), thumb };
}

function diff(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
  return d / a.length;
}

/** Choose up to `k` frames that cover the clip, are usable, and are not near-duplicates. Pure; tested. */
export function selectFrames<T extends FrameScore>(cands: T[], k = MAX_FRAMES): T[] {
  if (!cands.length) return [];
  const sharp = [...cands.map((c) => c.sharpness)].sort((a, b) => a - b);
  const median = sharp[Math.floor(sharp.length / 2)];
  const usable = cands.filter((c) => c.brightness >= MIN_BRIGHTNESS && c.brightness <= MAX_BRIGHTNESS
    && c.sharpness >= 0.5 * median);
  if (!usable.length) return [];
  const ordered = [...cands].sort((a, b) => a.time - b.time);
  const steps = ordered.slice(1).map((c, i) => diff(ordered[i].thumb, c.thumb)).sort((a, b) => a - b);
  const typicalMotion = steps.length ? steps[Math.floor(steps.length / 2)] : 0;
  const duplicateBelow = Math.max(STILL_DIFF, 0.25 * typicalMotion);
  const t0 = Math.min(...cands.map((c) => c.time));
  const t1 = Math.max(...cands.map((c) => c.time));
  const span = Math.max(t1 - t0, 1e-6);
  const picks: T[] = [];
  for (let s = 0; s < k; s++) {
    const lo = t0 + (s * span) / k;
    const hi = s === k - 1 ? Infinity : t0 + ((s + 1) * span) / k;
    const inSeg = usable.filter((c) => c.time >= lo && c.time < hi);
    if (!inSeg.length) continue;
    const best = inSeg.reduce((a, b) => (b.sharpness > a.sharpness ? b : a));
    const prev = picks[picks.length - 1];
    if (!prev || diff(prev.thumb, best.thumb) > duplicateBelow) picks.push(best);
  }
  return picks;
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode failed"))), "image/jpeg", 0.9));
}

/** Draw a video's current frame onto `canvas`, long side <= 1280 px (the server keeps <= 1024 anyway). */
export function drawFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => { video.removeEventListener("seeked", done); video.removeEventListener("error", fail); resolve(); };
    const fail = () => { video.removeEventListener("seeked", done); video.removeEventListener("error", fail); reject(new Error("seek failed")); };
    video.addEventListener("seeked", done);
    video.addEventListener("error", fail);
    video.currentTime = t;
  });
}

/** Sample an uploaded clip (first 30 s at most, up to 60 evenly spaced frames), then render the chosen ones. */
export async function sampleClip(file: File, onProgress?: (done: number, total: number) => void): Promise<VideoSample> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;
  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("This video format cannot be decoded by the browser."));
    });
    const duration = Math.min(Number.isFinite(video.duration) ? video.duration : 0, 30);
    if (!duration) throw new Error("The video has no playable duration.");
    const total = Math.min(60, Math.max(6, Math.ceil(duration * 4)));
    const canvas = document.createElement("canvas");
    const work = document.createElement("canvas");
    const scores: FrameScore[] = [];
    for (let i = 0; i < total; i++) {
      const t = ((i + 0.5) * duration) / total;
      await seek(video, t);
      drawFrame(video, canvas);
      scores.push(scoreCanvas(canvas, work, Number(t.toFixed(2))));
      onProgress?.(i + 1, total);
    }
    const chosen = selectFrames(scores);
    const frames: SampledFrame[] = [];
    for (const c of chosen) {
      await seek(video, c.time);
      drawFrame(video, canvas);
      const blob = await toBlob(canvas);
      frames.push({ ...c, blob, url: URL.createObjectURL(blob) });
    }
    return { source: "clip", duration: Number(duration.toFixed(2)), considered: total, frames };
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Live camera capture: while recording, grab a frame every 250 ms (keeping the JPEG of each candidate), then
 * select. Stops by itself after `maxSeconds`.
 */
export class CameraRecorder {
  private canvas = document.createElement("canvas");
  private work = document.createElement("canvas");
  private cands: SampledFrame[] = [];
  private timer: number | null = null;
  private t0 = 0;
  private busy = false;

  constructor(private video: HTMLVideoElement, private onTick: (count: number, seconds: number) => void,
    private maxSeconds = 20, private onAutoStop?: () => void) {}

  start() {
    this.cands.forEach((c) => URL.revokeObjectURL(c.url));
    this.cands = [];
    this.t0 = performance.now();
    this.timer = window.setInterval(() => void this.grab(), 250);
  }

  private async grab() {
    if (this.busy || !this.video.videoWidth) return;
    this.busy = true;
    try {
      const t = (performance.now() - this.t0) / 1000;
      drawFrame(this.video, this.canvas);
      const score = scoreCanvas(this.canvas, this.work, Number(t.toFixed(2)));
      const blob = await toBlob(this.canvas);
      this.cands.push({ ...score, blob, url: URL.createObjectURL(blob) });
      this.onTick(this.cands.length, t);
      if (t >= this.maxSeconds) this.onAutoStop?.();
    } finally {
      this.busy = false;
    }
  }

  stop(): VideoSample {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    const duration = (performance.now() - this.t0) / 1000;
    const chosen = selectFrames(this.cands);
    const keep = new Set(chosen);
    this.cands.filter((c) => !keep.has(c)).forEach((c) => URL.revokeObjectURL(c.url));
    const considered = this.cands.length;
    this.cands = [];
    return { source: "camera", duration: Number(duration.toFixed(2)), considered, frames: chosen };
  }
}
