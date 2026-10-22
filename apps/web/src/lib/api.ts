/**
 * Browser-side API access. Everything goes through the same-origin BFF (/api/...); the browser never sees
 * the internal API address or the session token (HttpOnly cookie).
 */

export class ApiError extends Error {
  status: number;
  code: string;
  requestId: string | null;
  retryable: boolean;
  details: Record<string, unknown>;

  constructor(status: number, code: string, message: string, requestId: string | null = null,
    retryable = false, details: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.retryable = retryable;
    this.details = details;
  }
}

/** API media URLs come back as "/v1/media/<token>"; the browser reaches them through the BFF at /api/v1/... */
export function mediaUrl(u: string | null | undefined): string | null {
  if (!u) return null;
  if (u.startsWith("/api/")) return u;
  if (u.startsWith("/v1/")) return `/api${u}`;
  return null; // never point the browser at an unexpected origin
}

async function parse(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function toError(status: number, body: unknown): ApiError {
  const e = (body as { error?: Record<string, unknown> } | null)?.error;
  if (e && typeof e === "object") {
    return new ApiError(status, String(e.code ?? "error"), String(e.message ?? "Request failed"),
      (e.request_id as string | null) ?? null, Boolean(e.retryable), (e.details as Record<string, unknown>) ?? {});
  }
  return new ApiError(status, "http_" + status, `Request failed with status ${status}`, null, status >= 500);
}

export interface FetchOptions extends Omit<RequestInit, "body"> {
  json?: unknown;
  body?: BodyInit | null;
  idempotencyKey?: string;
}

export async function api<T>(path: string, opts: FetchOptions = {}): Promise<T> {
  const headers = new Headers(opts.headers);
  let body: BodyInit | null | undefined = opts.body;
  if (opts.json !== undefined) {
    headers.set("content-type", "application/json");
    body = JSON.stringify(opts.json);
  }
  if (opts.idempotencyKey) headers.set("Idempotency-Key", opts.idempotencyKey);
  headers.set("X-Correlation-ID", newId());
  const url = path.startsWith("/api/") ? path : `/api${path.startsWith("/") ? "" : "/"}${path}`;
  let res: Response;
  try {
    res = await fetch(url, { ...opts, headers, body, credentials: "same-origin", cache: "no-store" });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the TreadOps web server.", null, true);
  }
  const data = await parse(res);
  if (!res.ok) {
    const err = toError(res.status, data);
    if (res.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("treadops:unauthorized", { detail: err.code }));
    }
    throw err;
  }
  return data as T;
}

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

/** Upload with progress (fetch cannot report upload progress). Resolves with parsed JSON or throws ApiError. */
export function uploadWithProgress<T>(url: string, form: FormData, opts: { idempotencyKey?: string;
  onProgress?: (fraction: number) => void } = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.withCredentials = true;
    if (opts.idempotencyKey) xhr.setRequestHeader("Idempotency-Key", opts.idempotencyKey);
    xhr.setRequestHeader("X-Correlation-ID", newId());
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onProgress?.(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "network_error", "Upload failed: network error.", null, true));
    xhr.onload = () => {
      let data: unknown = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        data = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as T);
      else {
        if (xhr.status === 401) window.dispatchEvent(new CustomEvent("treadops:unauthorized"));
        reject(toError(xhr.status, data));
      }
    };
    xhr.send(form);
  });
}
