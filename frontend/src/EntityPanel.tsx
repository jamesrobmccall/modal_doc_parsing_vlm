import { useEffect, useState } from "react";

type ExtractionMode = "per_page" | "whole_document";
type ExtractionPhase = "idle" | "suggesting" | "editing" | "extracting" | "done";
type InputMode = "document" | "text";

interface EntityFieldDefinition {
  name: string;
  field_type: string;
  description: string;
  required: boolean;
  examples: string[];
}

interface EntityDefinition {
  entity_name: string;
  description: string;
  fields: EntityFieldDefinition[];
}

interface ExtractedEntity {
  entity_name: string;
  page_id: number | null;
  data: Record<string, unknown>;
  confidence: number | null;
}

interface ExtractionResult {
  job_id: string;
  entities: ExtractedEntity[];
  schema_used: EntityDefinition[];
  extraction_mode: ExtractionMode;
  model_id: string;
  inference_ms: number;
}

const FIELD_TYPES = ["string", "number", "date", "boolean", "list[string]"];

function emptyField(): EntityFieldDefinition {
  return { name: "", field_type: "string", description: "", required: true, examples: [] };
}

function emptyEntity(): EntityDefinition {
  return { entity_name: "", description: "", fields: [emptyField()] };
}

export default function EntityPanel({ jobId }: { jobId: string | null }) {
  const [inputMode, setInputMode] = useState<InputMode>("document");
  const [rawText, setRawText] = useState("");
  const [isCreatingTextJob, setIsCreatingTextJob] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(jobId);

  const [phase, setPhase] = useState<ExtractionPhase>("idle");
  const [entities, setEntities] = useState<EntityDefinition[]>([]);
  const [docSummary, setDocSummary] = useState("");
  const [extractionMode, setExtractionMode] = useState<ExtractionMode>("whole_document");
  const [extractionResult, setExtractionResult] = useState<ExtractionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Sync activeJobId from prop when in document mode
  useEffect(() => {
    if (inputMode === "document") {
      setActiveJobId(jobId);
    }
  }, [jobId, inputMode]);

  // Reset state when switching input modes
  const switchInputMode = (mode: InputMode) => {
    setInputMode(mode);
    setPhase("idle");
    setEntities([]);
    setDocSummary("");
    setExtractionResult(null);
    setError(null);
    if (mode === "document") {
      setActiveJobId(jobId);
      setExtractionMode("per_page");
    } else {
      setActiveJobId(null);
      setExtractionMode("whole_document");
    }
  };

  const createTextJob = async () => {
    if (!rawText.trim()) return;
    setIsCreatingTextJob(true);
    setError(null);
    try {
      const res = await fetch("/api/text-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail || `Failed to create text job (${res.status})`);
      }
      const data = await res.json();
      setActiveJobId(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create text job.");
    } finally {
      setIsCreatingTextJob(false);
    }
  };

  const suggestEntities = async () => {
    if (!activeJobId) return;
    setPhase("suggesting");
    setError(null);
    try {
      const res = await fetch(`/api/jobs/${activeJobId}/entities/suggest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail || `Suggest failed (${res.status})`);
      }
      const data = await res.json();
      setEntities(data.suggested_entities ?? []);
      setDocSummary(data.document_summary ?? "");
      setPhase("editing");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to suggest entities.");
      setPhase("idle");
    }
  };

  const runExtraction = async () => {
    if (!activeJobId) return;
    setPhase("extracting");
    setError(null);
    try {
      const res = await fetch(`/api/jobs/${activeJobId}/entities/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: activeJobId, entities, extraction_mode: extractionMode }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail || `Extract failed (${res.status})`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start extraction.");
      setPhase("editing");
    }
  };

  // Poll for extraction result
  useEffect(() => {
    if (phase !== "extracting" || !activeJobId) return;
    let cancelled = false;
    let timeoutId: number | undefined;

    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${activeJobId}/entities/result`);
        if (res.status === 409) {
          if (!cancelled) timeoutId = window.setTimeout(poll, 3000);
          return;
        }
        if (res.status === 404) {
          if (!cancelled) timeoutId = window.setTimeout(poll, 3000);
          return;
        }
        if (!res.ok) throw new Error(`Result fetch failed (${res.status})`);
        const data = (await res.json()) as ExtractionResult;
        if (!cancelled) {
          setExtractionResult(data);
          setPhase("done");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch result.");
          setPhase("editing");
        }
      }
    };

    timeoutId = window.setTimeout(poll, 2000);
    return () => {
      cancelled = true;
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [phase, activeJobId]);

  const updateEntity = (idx: number, patch: Partial<EntityDefinition>) => {
    setEntities((prev) => prev.map((e, i) => (i === idx ? { ...e, ...patch } : e)));
  };

  const updateField = (entityIdx: number, fieldIdx: number, patch: Partial<EntityFieldDefinition>) => {
    setEntities((prev) =>
      prev.map((e, i) =>
        i === entityIdx
          ? { ...e, fields: e.fields.map((f, j) => (j === fieldIdx ? { ...f, ...patch } : f)) }
          : e
      )
    );
  };

  const removeField = (entityIdx: number, fieldIdx: number) => {
    setEntities((prev) =>
      prev.map((e, i) =>
        i === entityIdx ? { ...e, fields: e.fields.filter((_, j) => j !== fieldIdx) } : e
      )
    );
  };

  const addField = (entityIdx: number) => {
    setEntities((prev) =>
      prev.map((e, i) => (i === entityIdx ? { ...e, fields: [...e.fields, emptyField()] } : e))
    );
  };

  const removeEntity = (idx: number) => {
    setEntities((prev) => prev.filter((_, i) => i !== idx));
  };

  const addEntity = () => {
    setEntities((prev) => [...prev, emptyEntity()]);
  };

  const downloadResult = () => {
    if (!extractionResult) return;
    const blob = new Blob([JSON.stringify(extractionResult, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeJobId}-entities.json`;
    document.body.append(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const resetToEdit = () => {
    setExtractionResult(null);
    setPhase("editing");
  };

  // Group extracted entities by entity_name
  const groupedResults: Record<string, ExtractedEntity[]> = {};
  if (extractionResult) {
    for (const ent of extractionResult.entities) {
      (groupedResults[ent.entity_name] ??= []).push(ent);
    }
  }

  const canSuggest = inputMode === "document" ? !!activeJobId : !!activeJobId;
  const documentModeNoJob = inputMode === "document" && !activeJobId;

  return (
    <section className="entity-panel panel">
      <div className="panel-head">
        <h2>Entity Extraction</h2>
        <span>
          {phase === "idle" && (inputMode === "text" && !activeJobId ? "Paste text to extract entities" : "Ready to suggest entities")}
          {phase === "suggesting" && "Analyzing document..."}
          {phase === "editing" && `${entities.length} entit${entities.length === 1 ? "y" : "ies"} to extract`}
          {phase === "extracting" && "Running extraction on GPU..."}
          {phase === "done" && `${extractionResult?.entities.length ?? 0} results extracted`}
        </span>
      </div>

      <div className="panel-body entity-body">

        {/* Input mode switcher */}
        <div className="input-mode-toggle">
          <button
            type="button"
            className={`input-mode-btn${inputMode === "document" ? " active" : ""}`}
            onClick={() => switchInputMode("document")}
          >
            From Document
          </button>
          <button
            type="button"
            className={`input-mode-btn${inputMode === "text" ? " active" : ""}`}
            onClick={() => switchInputMode("text")}
          >
            Raw Text
          </button>
        </div>

        {/* Text input mode — show textarea until job is created */}
        {inputMode === "text" && !activeJobId && phase === "idle" && (
          <div className="text-input-mode">
            <p className="text-input-hint">
              Paste raw text below. A temporary job will be created so you can run entity
              suggestion and extraction directly, without parsing a document.
            </p>
            <textarea
              className="raw-text-input"
              value={rawText}
              onChange={(e) => setRawText(e.target.value)}
              placeholder="Paste your text here..."
              rows={8}
            />
            <button
              className="submit-btn"
              type="button"
              onClick={createTextJob}
              disabled={!rawText.trim() || isCreatingTextJob}
            >
              {isCreatingTextJob ? "Creating job..." : "Use This Text"}
            </button>
          </div>
        )}

        {/* Document mode — no job yet */}
        {documentModeNoJob && phase === "idle" && (
          <div className="entity-idle entity-idle--disabled">
            <p>
              Parse a document above to enable entity suggestion, or switch to{" "}
              <button
                type="button"
                className="inline-link"
                onClick={() => switchInputMode("text")}
              >
                Raw Text mode
              </button>{" "}
              to paste text directly.
            </p>
          </div>
        )}

        {/* Main idle state — job ready */}
        {phase === "idle" && canSuggest && (
          <div className="entity-idle">
            <p>
              {inputMode === "text"
                ? "Text loaded. Use the extraction model to suggest structured entities, then review and run extraction."
                : "Use the extraction model to suggest structured entities from your parsed document, then review and run extraction."}
            </p>
            <div className="entity-idle-actions">
              <button className="submit-btn" type="button" onClick={suggestEntities}>
                Suggest Entities
              </button>
              <button
                className="ghost-btn"
                type="button"
                onClick={() => {
                  setEntities([emptyEntity()]);
                  setPhase("editing");
                }}
              >
                Define Manually
              </button>
            </div>
          </div>
        )}

        {phase === "suggesting" && (
          <div className="entity-loading">
            <p>Analyzing document to suggest entities...</p>
            <div className="progress-track">
              <div className="progress-fill is-refining" style={{ width: "60%" }} />
            </div>
          </div>
        )}

        {phase === "editing" && (
          <div className="entity-editor">
            {docSummary && <p className="entity-summary">{docSummary}</p>}

            {inputMode === "document" && (
              <div className="entity-mode-toggle">
                <label className="field">
                  <span>Extraction Mode</span>
                  <select
                    value={extractionMode}
                    onChange={(e) => setExtractionMode(e.target.value as ExtractionMode)}
                  >
                    <option value="per_page">Per Page (extract from each page independently)</option>
                    <option value="whole_document">Whole Document (single extraction pass)</option>
                  </select>
                </label>
              </div>
            )}

            {entities.map((entity, entityIdx) => (
              <div key={entityIdx} className="entity-card">
                <div className="entity-card-header">
                  <input
                    className="entity-name-input"
                    value={entity.entity_name}
                    onChange={(e) => updateEntity(entityIdx, { entity_name: e.target.value })}
                    placeholder="Entity name (e.g. Invoice)"
                  />
                  <button
                    className="ghost-btn entity-remove-btn"
                    type="button"
                    onClick={() => removeEntity(entityIdx)}
                  >
                    Remove
                  </button>
                </div>
                <input
                  className="entity-desc-input"
                  value={entity.description}
                  onChange={(e) => updateEntity(entityIdx, { description: e.target.value })}
                  placeholder="Description"
                />

                <div className="entity-fields">
                  <div className="entity-field-header">
                    <span>Field</span>
                    <span>Type</span>
                    <span>Description</span>
                    <span>Req</span>
                    <span></span>
                  </div>
                  {entity.fields.map((field, fieldIdx) => (
                    <div key={fieldIdx} className="entity-field-row">
                      <input
                        value={field.name}
                        onChange={(e) => updateField(entityIdx, fieldIdx, { name: e.target.value })}
                        placeholder="field_name"
                      />
                      <select
                        value={field.field_type}
                        onChange={(e) => updateField(entityIdx, fieldIdx, { field_type: e.target.value })}
                      >
                        {FIELD_TYPES.map((t) => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                      <input
                        value={field.description}
                        onChange={(e) => updateField(entityIdx, fieldIdx, { description: e.target.value })}
                        placeholder="Description"
                      />
                      <input
                        type="checkbox"
                        checked={field.required}
                        onChange={(e) => updateField(entityIdx, fieldIdx, { required: e.target.checked })}
                      />
                      <button
                        className="ghost-btn entity-field-remove"
                        type="button"
                        onClick={() => removeField(entityIdx, fieldIdx)}
                      >
                        x
                      </button>
                    </div>
                  ))}
                  <button className="ghost-btn entity-add-field" type="button" onClick={() => addField(entityIdx)}>
                    + Add Field
                  </button>
                </div>
              </div>
            ))}

            <div className="entity-editor-actions">
              <button className="ghost-btn" type="button" onClick={addEntity}>
                + Add Entity
              </button>
              <button
                className="submit-btn"
                type="button"
                onClick={runExtraction}
                disabled={entities.length === 0 || entities.some((e) => !e.entity_name.trim())}
              >
                Run Extraction
              </button>
            </div>
          </div>
        )}

        {phase === "extracting" && (
          <div className="entity-loading">
            <p>Extracting entities using GPU inference with structured outputs...</p>
            <div className="progress-track">
              <div className="progress-fill is-refining" style={{ width: "40%" }} />
            </div>
          </div>
        )}

        {phase === "done" && extractionResult && (
          <div className="entity-results">
            <div className="entity-results-meta">
              <span>Model: {extractionResult.model_id}</span>
              <span>Mode: {extractionResult.extraction_mode}</span>
              <span>Inference: {(extractionResult.inference_ms / 1000).toFixed(1)}s</span>
            </div>

            {Object.entries(groupedResults).map(([entityName, items]) => (
              <div key={entityName} className="entity-result-table">
                <h3>{entityName}</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        {extractionResult.extraction_mode === "per_page" && <th>Page</th>}
                        {Object.keys(items[0]?.data ?? {}).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((item, idx) => (
                        <tr key={idx}>
                          {extractionResult.extraction_mode === "per_page" && (
                            <td>{item.page_id ?? "-"}</td>
                          )}
                          {Object.values(item.data).map((val, ci) => (
                            <td key={ci}>
                              {val === null ? <em className="null-value">null</em> : String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}

            <div className="entity-results-actions">
              <button className="submit-btn" type="button" onClick={downloadResult}>
                Download Entities (JSON)
              </button>
              <button className="ghost-btn" type="button" onClick={resetToEdit}>
                Edit &amp; Re-extract
              </button>
            </div>
          </div>
        )}

        {error && <p className="error-message">{error}</p>}
      </div>
    </section>
  );
}
