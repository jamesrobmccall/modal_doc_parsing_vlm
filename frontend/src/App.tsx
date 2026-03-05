import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import EntityPanel from "./EntityPanel";

type ParseMode = "balanced" | "accurate";
type LatencyProfile = "fast" | "balanced" | "max_quality";
type ProcessingProfile = "fast" | "balanced" | "max_quality";
type ResultFormat = "markdown" | "json" | "text";
type MarkdownView = "rendered" | "raw";

type JobStatus =
  | "queued"
  | "splitting"
  | "submitting"
  | "running"
  | "aggregating"
  | "completed_fast"
  | "completed_final"
  | "completed"
  | "completed_with_errors"
  | "failed";

interface PageErrorSummary {
  page_id: number;
  code: string;
  message: string;
  stage?: string | null;
  retry_count: number;
}

interface JobStatusPayload {
  status: JobStatus;
  pages_total: number;
  pages_completed: number;
  pages_running: number;
  pages_failed: number;
  progress_percent: number;
  timings: {
    split_ms: number;
    submit_ms: number;
    aggregate_ms: number;
    refine_ms?: number;
    elapsed_ms: number;
  };
  error_summary?: PageErrorSummary[] | null;
  result_revision: number;
  pending_refinement_pages: number;
}

interface CreateJobResponse {
  job_id: string;
  status: "queued";
  pages_total: number | null;
  poll_after_seconds: number;
  source_preview_url: string;
  mime_type: string;
}

interface ResultEnvelope {
  result: string | Record<string, unknown>;
}

const DEFAULT_POLL_INTERVAL_MS = 2000;
const RESULT_LEVEL_LATEST = "latest";
const STAGE_SEQUENCE: JobStatus[] = [
  "queued",
  "splitting",
  "submitting",
  "running",
  "aggregating",
  "completed_fast",
  "completed_final"
];
const TERMINAL_STATUSES = new Set<JobStatus>([
  "completed_final",
  "completed",
  "completed_with_errors",
  "failed"
]);
const RESULT_AVAILABLE_STATUSES = new Set<JobStatus>([
  "completed_fast",
  "completed_final",
  "completed",
  "completed_with_errors"
]);
const PROFILE_CONFIG: Record<
  ProcessingProfile,
  {
    label: string;
    eta: string;
    detail: string;
    mode: ParseMode;
    latencyProfile: LatencyProfile;
  }
> = {
  fast: {
    label: "Fast",
    eta: "~15-45s for short docs",
    detail: "Rapid GPU OCR pass for quick turn-around.",
    mode: "balanced",
    latencyProfile: "fast"
  },
  balanced: {
    label: "Balanced",
    eta: "~30-90s for short docs",
    detail: "Best default quality/speed blend with selective refinement.",
    mode: "balanced",
    latencyProfile: "balanced"
  },
  max_quality: {
    label: "Max Quality",
    eta: "~60-180s for short docs",
    detail: "Highest fidelity with richer prompts and deeper fallback work.",
    mode: "accurate",
    latencyProfile: "max_quality"
  }
};
const PROFILE_ORDER: ProcessingProfile[] = ["fast", "balanced", "max_quality"];

function stageLabel(status: JobStatus): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "splitting":
      return "Preparing pages";
    case "submitting":
      return "Dispatching tasks";
    case "running":
      return "OCR in progress";
    case "aggregating":
      return "Compiling output";
    case "completed_fast":
      return "Fast draft ready";
    case "completed_final":
      return "Final result ready";
    case "completed":
      return "Completed";
    case "completed_with_errors":
      return "Completed with errors";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

function shouldPoll(statusPayload: JobStatusPayload | null): boolean {
  if (statusPayload == null) {
    return true;
  }
  if (
    statusPayload.status === "completed_fast" &&
    statusPayload.pending_refinement_pages > 0
  ) {
    return true;
  }
  return !TERMINAL_STATUSES.has(statusPayload.status);
}

function isImageMime(mimeType: string | null): boolean {
  return mimeType === "image/png" || mimeType === "image/jpeg";
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

function normalizeTimelineStatus(status: JobStatus): JobStatus {
  if (status === "completed" || status === "completed_with_errors") {
    return "completed_final";
  }
  return status;
}

function stageDetail(status: JobStatus): string {
  switch (status) {
    case "queued":
      return "Your job is waiting for available workers.";
    case "splitting":
      return "Document pages are being rasterized and prepared.";
    case "submitting":
      return "Page tasks are being dispatched to GPU OCR workers.";
    case "running":
      return "GPU-accelerated OCR is extracting layout and text.";
    case "aggregating":
      return "Page outputs are being merged into document Markdown.";
    case "completed_fast":
      return "Fast draft is ready; final refinement may still complete.";
    case "completed_final":
    case "completed":
      return "Final Markdown is ready for preview and download.";
    case "completed_with_errors":
      return "Completed with partial errors; review the error summary.";
    case "failed":
      return "The parse failed. Try again or use a different profile.";
    default:
      return "";
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  const fallback = `Request failed with ${response.status}.`;
  try {
    const payload = (await response.json()) as { detail?: string; message?: string };
    return payload.detail ?? payload.message ?? fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [processingProfile, setProcessingProfile] =
    useState<ProcessingProfile>("balanced");
  const [markdownView, setMarkdownView] = useState<MarkdownView>("rendered");

  const [jobId, setJobId] = useState<string | null>(null);
  const [sourcePreviewUrl, setSourcePreviewUrl] = useState<string | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [sourceMimeType, setSourceMimeType] = useState<string | null>(null);
  const [pollIntervalMs, setPollIntervalMs] = useState(DEFAULT_POLL_INTERVAL_MS);

  const [statusPayload, setStatusPayload] = useState<JobStatusPayload | null>(null);
  const [markdownResult, setMarkdownResult] = useState<string>("");

  const [formError, setFormError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const initialJobId = params.get("job");
    const initialMime = params.get("mime");

    if (initialJobId) {
      setJobId(initialJobId);
      setSourcePreviewUrl(`/api/jobs/${initialJobId}/source`);
    }
    if (initialMime) {
      setSourceMimeType(initialMime);
    }
  }, []);

  useEffect(() => {
    if (!file) {
      setLocalPreviewUrl(null);
      return;
    }

    const next = URL.createObjectURL(file);
    setLocalPreviewUrl(next);
    return () => {
      URL.revokeObjectURL(next);
    };
  }, [file]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (jobId) {
      params.set("job", jobId);
      if (sourceMimeType) {
        params.set("mime", sourceMimeType);
      }
    } else {
      params.delete("job");
      params.delete("mime");
    }
    params.delete("level");

    const query = params.toString();
    const nextUrl = query
      ? `${window.location.pathname}?${query}`
      : window.location.pathname;
    window.history.replaceState({}, "", nextUrl);
  }, [jobId, sourceMimeType]);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    let cancelled = false;
    let timeoutId: number | undefined;

    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}/status`);
        if (!response.ok) {
          throw new Error(await readErrorMessage(response));
        }

        const payload = (await response.json()) as JobStatusPayload;
        if (cancelled) {
          return;
        }

        setStatusPayload(payload);
        if (shouldPoll(payload)) {
          timeoutId = window.setTimeout(pollStatus, pollIntervalMs);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setFormError(
          error instanceof Error ? error.message : "Unable to poll job status."
        );
        timeoutId = window.setTimeout(pollStatus, pollIntervalMs);
      }
    };

    void pollStatus();

    return () => {
      cancelled = true;
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [jobId, pollIntervalMs]);

  useEffect(() => {
    if (!jobId || !statusPayload) {
      return;
    }
    if (!RESULT_AVAILABLE_STATUSES.has(statusPayload.status)) {
      return;
    }

    let cancelled = false;
    let retryId: number | undefined;

    const fetchMarkdown = async () => {
      const response = await fetch(
        `/api/jobs/${jobId}/result?format=markdown&result_level=${RESULT_LEVEL_LATEST}`
      );
      if (response.status === 409) {
        setMarkdownResult("");
        setActionMessage("Result is still processing for this job.");
        if (!cancelled) {
          retryId = window.setTimeout(() => {
            void fetchMarkdown();
          }, 1500);
        }
        return;
      }
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = (await response.json()) as ResultEnvelope;
      if (cancelled) {
        return;
      }
      setMarkdownResult(typeof payload.result === "string" ? payload.result : "");
    };

    void fetchMarkdown().catch((error) => {
      if (cancelled) {
        return;
      }
      setFormError(error instanceof Error ? error.message : "Unable to load markdown.");
    });

    return () => {
      cancelled = true;
      if (retryId !== undefined) {
        window.clearTimeout(retryId);
      }
    };
  }, [jobId, statusPayload?.status, statusPayload?.result_revision]);

  const statusText = useMemo(() => {
    if (!statusPayload) {
      return "Upload a document to begin.";
    }
    if (
      statusPayload.status === "completed_fast" &&
      statusPayload.pending_refinement_pages > 0
    ) {
      return `Fast result is ready. Refining ${statusPayload.pending_refinement_pages} page${statusPayload.pending_refinement_pages === 1 ? "" : "s"} for final quality.`;
    }
    return stageLabel(statusPayload.status);
  }, [statusPayload]);
  const statusDetailText = useMemo(() => {
    if (!statusPayload) {
      return "Pipeline: queue -> split pages -> dispatch GPU OCR -> aggregate -> final Markdown.";
    }
    return stageDetail(statusPayload.status);
  }, [statusPayload]);

  const progressPercent = useMemo(() => {
    if (!statusPayload) {
      return 0;
    }
    return Math.max(0, Math.min(100, statusPayload.progress_percent));
  }, [statusPayload]);

  const timelineStatus = normalizeTimelineStatus(statusPayload?.status ?? "queued");
  const timelineIndex = STAGE_SEQUENCE.indexOf(timelineStatus);

  const setFileFromInput = (nextFile: File | null) => {
    setFile(nextFile);
    if (nextFile) {
      setSourcePreviewUrl(null);
      setSourceMimeType(nextFile.type || null);
      setActionMessage(`${nextFile.name} selected.`);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setFormError("Choose a PDF, PNG, or JPEG file first.");
      return;
    }

    setIsSubmitting(true);
    setFormError(null);
    setActionMessage(null);
    setMarkdownResult("");
    setStatusPayload(null);

    const profileConfig = PROFILE_CONFIG[processingProfile];
    const formData = new FormData();
    formData.set("file", file);
    formData.set("mode", profileConfig.mode);
    formData.set("latency_profile", profileConfig.latencyProfile);
    formData.set("result_level", RESULT_LEVEL_LATEST);

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const payload = (await response.json()) as CreateJobResponse;
      setJobId(payload.job_id);
      setSourcePreviewUrl(payload.source_preview_url);
      setSourceMimeType(payload.mime_type);
      setPollIntervalMs(
        Math.max(1, payload.poll_after_seconds) * 1000 || DEFAULT_POLL_INTERVAL_MS
      );
      setActionMessage(`Job ${payload.job_id.slice(0, 8)} started.`);
    } catch (error) {
      setFormError(
        error instanceof Error ? error.message : "Unable to start a parsing job."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const copyMarkdown = async () => {
    if (!markdownResult.trim()) {
      return;
    }
    try {
      await navigator.clipboard.writeText(markdownResult);
      setActionMessage("Markdown copied to clipboard.");
    } catch {
      setActionMessage("Clipboard write failed in this browser context.");
    }
  };

  const downloadResult = async (format: ResultFormat) => {
    if (!jobId) {
      return;
    }
    setFormError(null);

    try {
      const response = await fetch(
        `/api/jobs/${jobId}/result?format=${format}&result_level=${RESULT_LEVEL_LATEST}`
      );
      if (response.status === 409) {
        setActionMessage("Result is still processing. Try again shortly.");
        return;
      }
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const payload = (await response.json()) as ResultEnvelope;

      let body: string;
      let contentType: string;
      let extension: string;
      if (format === "json") {
        body = JSON.stringify(payload.result, null, 2);
        contentType = "application/json;charset=utf-8";
        extension = "json";
      } else if (format === "markdown") {
        body = typeof payload.result === "string" ? payload.result : "";
        contentType = "text/markdown;charset=utf-8";
        extension = "md";
      } else {
        body = typeof payload.result === "string" ? payload.result : "";
        contentType = "text/plain;charset=utf-8";
        extension = "txt";
      }

      const blob = new Blob([body], { type: contentType });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${jobId}.${extension}`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setActionMessage(`${format.toUpperCase()} download started.`);
    } catch (error) {
      setFormError(
        error instanceof Error ? error.message : "Unable to download result."
      );
    }
  };

  const clearSession = () => {
    setFile(null);
    setJobId(null);
    setStatusPayload(null);
    setSourcePreviewUrl(null);
    setSourceMimeType(null);
    setMarkdownResult("");
    setFormError(null);
    setActionMessage("Session cleared.");
  };
  const goHome = () => {
    window.location.assign(window.location.pathname);
  };

  const activeSourcePreviewUrl = sourcePreviewUrl ?? localPreviewUrl;
  const selectedProfile = PROFILE_CONFIG[processingProfile];

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-top-row">
          <p className="eyebrow">Modal-hosted GPU OCR + Markdown conversion</p>
          <button type="button" className="hero-home-btn" onClick={goHome}>
            Home / New Parse
          </button>
        </div>
        <h1>Parse documents. Extract entities.</h1>
        <p className="hero-copy">
          Upload a PDF or image for GPU-accelerated OCR to Markdown, or paste raw text directly
          to run structured entity extraction — with GLM-5 or Qwen 2.5.
        </p>
        <div className="hero-badges">
          <span>GPU workers</span>
          <span>Live stage telemetry</span>
          <span>GLM-5 &amp; Qwen entity extraction</span>
        </div>
      </header>

      <section className="workflows">
        <div className="parse-col">
        <form onSubmit={handleSubmit} className="upload-form">
          <label
            className={`dropzone ${isDragging ? "is-dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              setFileFromInput(event.dataTransfer.files?.[0] ?? null);
            }}
          >
            <span className="dropzone-title">Document</span>
            <span className="dropzone-copy">
              Drag and drop a PDF/PNG/JPEG, or click to choose a file.
            </span>
            <input
              type="file"
              aria-label="Document"
              accept="application/pdf,image/png,image/jpeg"
              onChange={(event) => setFileFromInput(event.target.files?.[0] ?? null)}
            />
            {file ? (
              <span className="file-chip">
                {file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
            ) : null}
          </label>

          <div className="field-grid">
            <label className="field">
              <span>OCR Profile</span>
              <select
                onChange={(event) =>
                  setProcessingProfile(event.target.value as ProcessingProfile)
                }
                value={processingProfile}
              >
                {PROFILE_ORDER.map((profile) => (
                  <option key={profile} value={profile}>
                    {PROFILE_CONFIG[profile].label} ({PROFILE_CONFIG[profile].eta})
                  </option>
                ))}
              </select>
              <span className="field-hint">{selectedProfile.detail}</span>
            </label>
            <div className="profile-guide" aria-label="OCR profile guide">
              {PROFILE_ORDER.map((profile) => (
                <div
                  key={profile}
                  className={`profile-guide-item ${
                    processingProfile === profile ? "is-selected" : ""
                  }`}
                >
                  <strong>
                    {PROFILE_CONFIG[profile].label} ({PROFILE_CONFIG[profile].eta})
                  </strong>
                  <p>{PROFILE_CONFIG[profile].detail}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="actions-row">
            <button className="submit-btn" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Submitting..." : "Parse Document"}
            </button>
            <button className="ghost-btn" type="button" onClick={clearSession}>
              Clear
            </button>
          </div>
        </form>

        <div className="status-panel" aria-live="polite">
          <div className="status-heading-row">
            <h2>Progress</h2>
            {statusPayload ? (
              <span className="status-chip">{stageLabel(statusPayload.status)}</span>
            ) : null}
          </div>
          <p className="status-text">{statusText}</p>
          <p className="status-detail">{statusDetailText}</p>
          <div
            className="progress-track"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progressPercent)}
          >
            <div
              className={`progress-fill ${
                statusPayload?.status === "completed_fast" &&
                statusPayload.pending_refinement_pages > 0
                  ? "is-refining"
                  : ""
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <ul className="timeline" aria-label="Processing timeline">
            {STAGE_SEQUENCE.map((stage, index) => {
              const state =
                statusPayload?.status === "failed"
                  ? "failed"
                  : index < timelineIndex
                    ? "done"
                    : index === timelineIndex
                      ? "active"
                      : "todo";
              return (
                <li key={stage} className={`timeline-item ${state}`}>
                  <span className="timeline-dot" />
                  {stageLabel(stage)}
                </li>
              );
            })}
          </ul>

          <div className="status-metrics">
            <div>
              <span>Completed</span>
              <strong>{statusPayload?.pages_completed ?? 0}</strong>
            </div>
            <div>
              <span>Running</span>
              <strong>{statusPayload?.pages_running ?? 0}</strong>
            </div>
            <div>
              <span>Failed</span>
              <strong>{statusPayload?.pages_failed ?? 0}</strong>
            </div>
            <div>
              <span>Total pages</span>
              <strong>{statusPayload?.pages_total ?? 0}</strong>
            </div>
            <div>
              <span>Elapsed</span>
              <strong>
                {statusPayload
                  ? formatDuration(statusPayload.timings.elapsed_ms)
                  : "-"}
              </strong>
            </div>
          </div>

          {statusPayload?.error_summary?.length ? (
            <div className="error-summary">
              <h3>Errors</h3>
              <ul>
                {statusPayload.error_summary.map((item, index) => (
                  <li key={`${item.page_id}-${index}`}>
                    page {item.page_id}: {item.code} - {item.message}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {jobId ? <p className="job-id">Job ID: {jobId}</p> : null}
          {actionMessage ? <p className="action-message">{actionMessage}</p> : null}
          {formError ? <p className="error-message">{formError}</p> : null}
        </div>
        </div>

        <EntityPanel jobId={jobId} />
      </section>

      <section className="workspace">
        <article className="panel source-panel">
          <div className="panel-head">
            <h2>Source Preview</h2>
            <span>{sourceMimeType ?? "No file loaded"}</span>
          </div>
          <div className="panel-body source-body">
            {activeSourcePreviewUrl ? (
              isImageMime(sourceMimeType) ? (
                <img
                  src={activeSourcePreviewUrl}
                  alt="Uploaded source"
                  className="source-image"
                />
              ) : (
                <iframe
                  title="Document preview"
                  src={activeSourcePreviewUrl}
                  className="source-frame"
                />
              )
            ) : (
              <p className="placeholder-copy">
                Upload a document to view the source preview.
              </p>
            )}
          </div>
        </article>

        <article className="panel markdown-panel">
          <div className="panel-head">
            <h2>Markdown Preview</h2>
            <div className="toolbar">
              <div className="toolbar-group">
                <span className="toolbar-label">Preview</span>
                <button
                  type="button"
                  onClick={() => setMarkdownView("rendered")}
                  className={markdownView === "rendered" ? "is-active" : ""}
                >
                  View rendered
                </button>
                <button
                  type="button"
                  onClick={() => setMarkdownView("raw")}
                  className={markdownView === "raw" ? "is-active" : ""}
                >
                  View raw
                </button>
              </div>
              <div className="toolbar-group">
                <span className="toolbar-label">Actions</span>
                <button type="button" onClick={copyMarkdown} disabled={!markdownResult}>
                  Copy markdown
                </button>
              </div>
              <div className="toolbar-group">
                <span className="toolbar-label">Download</span>
                <button
                  type="button"
                  onClick={() => downloadResult("markdown")}
                  disabled={!jobId}
                >
                  Save .md
                </button>
                <button
                  type="button"
                  onClick={() => downloadResult("json")}
                  disabled={!jobId}
                >
                  Save .json
                </button>
                <button
                  type="button"
                  onClick={() => downloadResult("text")}
                  disabled={!jobId}
                >
                  Save .txt
                </button>
              </div>
            </div>
          </div>
          <div className="panel-body markdown-body">
            {markdownResult ? (
              markdownView === "rendered" ? (
                <div className="markdown-rendered">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownResult}</ReactMarkdown>
                </div>
              ) : (
                <pre className="markdown-raw">{markdownResult}</pre>
              )
            ) : (
              <p className="placeholder-copy">
                Markdown output appears as soon as processing produces a result.
              </p>
            )}
          </div>
        </article>
      </section>

    </div>
  );
}
