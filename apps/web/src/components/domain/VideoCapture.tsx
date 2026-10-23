"use client";

import clsx from "clsx";
import { Circle, Loader2, Square, Video, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { CameraRecorder, MAX_FRAMES, sampleClip, type VideoSample } from "@/lib/video";

/** Video staged in the composer: a clip being analysed, or the live camera, and the frames chosen to send. */
export function useVideoPicker(blockedReason: string | null) {
  const [mode, setMode] = useState<null | "clip" | "camera">(null);
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<VideoSample | null>(null);
  const [error, setError] = useState<string | null>(null);
  const release = (s: VideoSample | null) => s?.frames.forEach((f) => URL.revokeObjectURL(f.url));
  const startClip = (f: File | null | undefined) => {
    setError(null);
    if (!f) return;
    if (blockedReason) return setError(blockedReason);
    if (!f.type.startsWith("video/")) return setError(`Unsupported file type ${f.type || "(unknown)"}; use an MP4, WebM or MOV clip.`);
    release(sample);
    setSample(null);
    setFile(f);
    setMode("clip");
  };
  const startCamera = () => {
    setError(null);
    if (blockedReason) return setError(blockedReason);
    release(sample);
    setSample(null);
    setFile(null);
    setMode("camera");
  };
  const clear = () => {
    release(sample);
    setMode(null);
    setFile(null);
    setSample(null);
    setError(null);
  };
  const removeFrame = (i: number) => setSample((s) => {
    if (!s) return s;
    URL.revokeObjectURL(s.frames[i].url);
    return { ...s, frames: s.frames.filter((_, j) => j !== i) };
  });
  return { mode, file, sample, error, blockedReason, startClip, startCamera, setSample, setError, clear, removeFrame,
    ready: Boolean(sample && sample.frames.length) };
}
export type VideoPicker = ReturnType<typeof useVideoPicker>;

function ClipAnalysis({ video }: { video: VideoPicker }) {
  const [progress, setProgress] = useState<[number, number]>([0, 0]);
  const { file, setSample, setError } = video;
  useEffect(() => {
    if (!file) return;
    let alive = true;
    sampleClip(file, (done, total) => { if (alive) setProgress([done, total]); })
      .then((s) => {
        if (!alive) return;
        if (s.frames.length) setSample(s);
        else setError("No usable frames: the clip is too dark, too blurred or does not move.");
      })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [file, setSample, setError]);
  const [done, total] = progress;
  return (
    <div className="flex items-center gap-3 text-sm text-ink-soft" aria-live="polite">
      <Loader2 aria-hidden className="h-4 w-4 animate-spin text-primary" />
      Analysing {file?.name}: {done}/{total || "…"} frames scored
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
        <div className="h-full bg-primary transition-[width]" style={{ width: total ? `${(done / total) * 100}%` : "0%" }} />
      </div>
    </div>
  );
}

function LiveCamera({ video }: { video: VideoPicker }) {
  const ref = useRef<HTMLVideoElement>(null);
  const recorder = useRef<CameraRecorder | null>(null);
  const [state, setState] = useState<"starting" | "ready" | "recording">("starting");
  const [tick, setTick] = useState<[number, number]>([0, 0]);
  const { setSample, setError } = video;
  useEffect(() => {
    let stream: MediaStream | null = null;
    let alive = true;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false })
      .then((s) => {
        stream = s;
        if (!alive || !ref.current) return s.getTracks().forEach((t) => t.stop());
        ref.current.srcObject = s;
        void ref.current.play();
        setState("ready");
      })
      .catch((e: Error) => { if (alive) setError(`Camera unavailable: ${e.message}`); });
    return () => { alive = false; stream?.getTracks().forEach((t) => t.stop()); };
  }, [setError]);
  const stop = () => {
    const s = recorder.current?.stop();
    recorder.current = null;
    if (!s) return;
    if (s.frames.length) setSample(s);
    else { setError("No usable frames: hold the camera steady in good light and turn the tyre slowly."); setState("ready"); }
  };
  const start = () => {
    if (!ref.current) return;
    recorder.current = new CameraRecorder(ref.current, (n, t) => setTick([n, t]), 20, stop);
    recorder.current.start();
    setState("recording");
  };
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-navy sm:w-72">
        <video ref={ref} muted playsInline aria-label="Live camera preview" className="h-full w-full object-cover" />
        {state === "recording" && (
          <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded-full bg-black/65 px-2.5 py-0.5 text-xs font-bold text-white">
            <span aria-hidden className="pulse-dot h-2 w-2 rounded-full bg-bad" /> REC {tick[1].toFixed(0)} s
          </span>
        )}
      </div>
      <div className="flex-1 space-y-2 text-sm text-ink-soft">
        <p>Point the camera at the tyre and turn it slowly through a full rotation. Recording stops by itself after 20 s.</p>
        {state === "recording" ? (
          <button type="button" onClick={stop} className="inline-flex items-center gap-2 rounded-lg bg-ink px-4 py-2 font-semibold text-white">
            <Square aria-hidden className="h-4 w-4" /> Stop ({tick[0]} frames)
          </button>
        ) : (
          <button type="button" onClick={start} disabled={state !== "ready"}
            className="inline-flex items-center gap-2 rounded-lg bg-bad px-4 py-2 font-semibold text-white disabled:opacity-50">
            <Circle aria-hidden className="h-4 w-4 fill-current" /> {state === "starting" ? "Starting camera…" : "Start recording"}
          </button>
        )}
      </div>
    </div>
  );
}

/** The panel above the composer while a video is being prepared or is ready to send. */
export function VideoPanel({ video }: { video: VideoPicker }) {
  if (!video.mode) return null;
  const s = video.sample;
  return (
    <div className="mb-2 space-y-2 rounded-xl bg-surface-muted p-3" aria-label="Video to send">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink">
        <Video aria-hidden className="h-4 w-4 text-primary" />
        {video.mode === "camera" ? "Live camera" : "Video clip"}
        {s && <span className="font-normal text-muted">· {s.frames.length} of {s.considered} frames over {s.duration.toFixed(1)} s</span>}
        <button type="button" aria-label="Remove video" onClick={video.clear} className="ml-auto rounded-lg p-1.5 text-muted hover:bg-line"><X className="h-4 w-4" /></button>
      </div>
      {!s && !video.error && (video.mode === "clip" ? <ClipAnalysis video={video} /> : <LiveCamera video={video} />)}
      {s && (
        <>
          <ol className="flex gap-2 overflow-x-auto pb-1" aria-label="Frames to send">
            {s.frames.map((f, i) => (
              <li key={f.url} className="relative shrink-0">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={f.url} alt={`Frame at ${f.time.toFixed(1)} s`} className="h-20 w-28 rounded-lg object-cover" />
                <span className="mono absolute bottom-1 left-1 rounded bg-black/65 px-1 text-[10px] text-white">{f.time.toFixed(1)} s</span>
                <button type="button" aria-label={`Drop frame at ${f.time.toFixed(1)} s`} onClick={() => video.removeFrame(i)}
                  className="absolute right-1 top-1 rounded-full bg-black/65 p-0.5 text-white hover:bg-bad"><X className="h-3 w-3" /></button>
              </li>
            ))}
          </ol>
          <p className={clsx("text-xs", s.frames.length ? "text-muted" : "text-bad")}>
            {s.frames.length
              ? `Sharpest well-exposed frame per sector of the turn (up to ${MAX_FRAMES}), duplicates dropped. Bib inspects each and decides per tyre.`
              : "All frames removed; record or attach again."}
          </p>
        </>
      )}
      {video.error && <p className="text-sm font-semibold text-bad" role="alert">{video.error}</p>}
    </div>
  );
}
