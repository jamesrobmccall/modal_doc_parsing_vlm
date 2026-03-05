import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, "", "/");
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("submits a file, polls status, and renders markdown output", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          job_id: "job-123",
          status: "queued",
          pages_total: 2,
          poll_after_seconds: 2,
          source_preview_url: "/api/jobs/job-123/source",
          mime_type: "application/pdf"
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          status: "completed_fast",
          pages_total: 2,
          pages_completed: 2,
          pages_running: 0,
          pages_failed: 0,
          progress_percent: 100,
          timings: {
            split_ms: 10,
            submit_ms: 20,
            aggregate_ms: 30,
            elapsed_ms: 80
          },
          result_revision: 1,
          pending_refinement_pages: 0
        })
      )
      .mockResolvedValueOnce(
        jsonResponse({
          result: "# Parsed Heading\n\nBody text"
        })
      );

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    const user = userEvent.setup();
    const fileInput = screen.getByLabelText("Document");
    await user.upload(fileInput, new File(["pdf"], "sample.pdf", { type: "application/pdf" }));
    await user.selectOptions(screen.getByRole("combobox"), "fast");
    expect(screen.getByTitle("Document preview")).toHaveAttribute(
      "src",
      "blob:preview-url"
    );
    await user.click(screen.getByRole("button", { name: "Parse Document" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/jobs",
        expect.objectContaining({ method: "POST" })
      );
    });
    const createCall = fetchMock.mock.calls.find((args) => args[0] === "/api/jobs");
    expect(createCall).toBeDefined();
    const createBody = createCall?.[1] as RequestInit;
    const formData = createBody.body as FormData;
    expect(formData.get("mode")).toBe("balanced");
    expect(formData.get("latency_profile")).toBe("fast");
    expect(formData.get("result_level")).toBe("latest");

    await waitFor(() => {
      expect(screen.getAllByText("Fast draft ready").length).toBeGreaterThan(0);
    });

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Parsed Heading" })
      ).toBeInTheDocument();
    });

    expect(screen.getByTitle("Document preview")).toBeInTheDocument();
  });

  it("hydrates from query params and shows running progress", async () => {
    window.history.replaceState({}, "", "/?job=abc123&mime=image/png");

    const fetchMock = vi.fn().mockResolvedValueOnce(
      jsonResponse({
        status: "running",
        pages_total: 4,
        pages_completed: 1,
        pages_running: 3,
        pages_failed: 0,
        progress_percent: 25,
        timings: {
          split_ms: 11,
          submit_ms: 12,
          aggregate_ms: 0,
          elapsed_ms: 90
        },
        result_revision: 0,
        pending_refinement_pages: 0
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith("/api/jobs/abc123/status");
    });

    expect(screen.getAllByText("OCR in progress").length).toBeGreaterThan(0);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByAltText("Uploaded source")).toBeInTheDocument();
  });
});
