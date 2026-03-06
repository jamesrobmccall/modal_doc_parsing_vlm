import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import EntityPanel from "./EntityPanel";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

describe("EntityPanel", () => {
  it("removes the backend toggle and suggests entities from a parsed document", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          job_id: "job-1",
          suggested_entities: [],
          document_summary: "summary"
        })
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<EntityPanel jobId="job-1" />);
    expect(screen.queryByText("Model Backend")).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Suggest Entities" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/jobs/job-1/entities/suggest",
        expect.objectContaining({ method: "POST" })
      );
    });
    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(request.body).toBe("{}");
  });

  it("still supports raw text flow without a backend toggle", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "text-job-1" }));
    vi.stubGlobal("fetch", fetchMock);

    render(<EntityPanel jobId={null} />);
    expect(screen.queryByText("Model Backend")).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Raw Text" }));
    await user.type(screen.getByPlaceholderText("Paste your text here..."), "Invoice 123");
    await user.click(screen.getByRole("button", { name: "Use This Text" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/text-jobs",
        expect.objectContaining({ method: "POST" })
      );
    });
  });
});
