import { useEffect, useState } from "react";
import {
  classify,
  generateGradCam,
  submitReview,
  type ClassificationResponse,
  type GradCamResponse,
} from "./api";
import "./styles.css";

function displayLabel(label: string) {
  return label.replaceAll("_", " ");
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [previewError, setPreviewError] = useState(false);
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [gradCam, setGradCam] = useState<GradCamResponse | null>(null);
  const [status, setStatus] = useState("Choose a chest X-ray image to begin.");
  const [busy, setBusy] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [reviewNotice, setReviewNotice] = useState("");

  useEffect(() => {
    setPreview("");
    setPreviewError(false);

    if (!file) return;

    const reader = new FileReader();
    let active = true;

    reader.onload = () => {
      if (active && typeof reader.result === "string") {
        setPreview(reader.result);
      }
    };

    reader.onerror = () => {
      if (active) setPreviewError(true);
    };

    reader.readAsDataURL(file);

    return () => {
      active = false;
      if (reader.readyState === FileReader.LOADING) reader.abort();
    };
  }, [file]);

  function selectFile(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    setGradCam(null);
    setHasError(false);
    setReviewNotice("");
    setStatus(
      nextFile
        ? "Image selected. Run classification to view model scores."
        : "Choose a chest X-ray image to begin.",
    );
  }

  async function handleUpload() {
    if (!file) {
      setHasError(true);
      setStatus("Select an image first.");
      return;
    }

    setBusy(true);
    setHasError(false);
    setResult(null);
    setGradCam(null);
    setReviewNotice("");
    setStatus("Running the educational classifier...");

    try {
      setResult(await classify(file));
      setStatus("Classification completed.");
    } catch (error) {
      setHasError(true);
      setStatus(
        error instanceof Error
          ? error.message
          : "The classification request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleGradCam(label: string) {
    if (!file || !result) return;

    setBusy(true);
    setHasError(false);
    setGradCam(null);
    setStatus(`Generating Grad-CAM for ${displayLabel(label)}...`);

    try {
      setGradCam(await generateGradCam(file, label));
      setStatus(`Grad-CAM generated for ${displayLabel(label)}.`);
    } catch (error) {
      setHasError(true);
      setStatus(
        error instanceof Error ? error.message : "Grad-CAM generation failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleReview(action: "approve" | "reject") {
    if (!result || !file) return;

    setBusy(true);
    setHasError(false);
    setReviewNotice("");
    setStatus("Saving reviewer feedback...");

    try {
      await submitReview(file.name, action, result.model_artifact);
      const notice =
        action === "approve"
          ? "Review approved successfully."
          : "Review rejected successfully.";
      setReviewNotice(notice);
      setStatus(notice);
    } catch (error) {
      setHasError(true);
      setReviewNotice("");
      setStatus(
        error instanceof Error ? error.message : "The review action failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  const predictions = result
    ? [...result.predictions].sort((a, b) => b.probability - a.probability)
    : [];

  return (
    <main>
      <header className="page-header">
        <div>
          <p className="eyebrow">EXPLAINABLE IMAGING WORKSPACE</p>
          <h1>AI-Assisted Medical Imaging</h1>
          <p className="subtitle">
            Chest X-ray classification, visual attribution, and reviewer feedback.
          </p>
        </div>
        <span className="badge">Educational prototype</span>
      </header>

      <div className="notice">
        Educational demonstration only — not for clinical use or diagnosis.
        Use only public or appropriately de-identified images.
      </div>

      <section className="panel upload-panel" aria-labelledby="upload-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / INPUT</p>
            <h2 id="upload-heading">Classification</h2>
          </div>
          <span className="muted">PNG · JPEG · BMP · TIFF</span>
        </div>

        <div className="upload-controls">
          <div className="file-control">
            <label htmlFor="image-upload">Chest X-ray image</label>
            <input
              id="image-upload"
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/tiff"
              disabled={busy}
              onChange={(event) =>
                selectFile(event.target.files?.[0] ?? null)
              }
            />
          </div>

          <button
            className="button-primary"
            type="button"
            onClick={handleUpload}
            disabled={busy || !file}
          >
            {busy ? "Working..." : "Run classification"}
          </button>
        </div>

        <p
          role="status"
          aria-live="polite"
          className={`status ${hasError ? "status-error" : ""}`}
        >
          {busy && <span className="spinner" aria-hidden="true" />}
          {status}
        </p>
      </section>

      <div className="workspace">
        <section className="panel image-panel" aria-labelledby="images-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">02 / VISUAL WORKSPACE</p>
              <h2 id="images-heading">Image & attribution</h2>
            </div>
          </div>

          <div className="image-grid">
            <figure>
              <figcaption>Original image</figcaption>
              <div className="image-frame">
                {preview && !previewError ? (
                  <img
                    src={preview}
                    alt="Uploaded chest X-ray"
                    onError={() => setPreviewError(true)}
                  />
                ) : (
                  <p className="image-placeholder">
                    {previewError
                      ? "Preview unavailable for this image format."
                      : "Select an image to preview it here."}
                  </p>
                )}
              </div>
              <p className="image-caption">{file?.name ?? "No image selected"}</p>
            </figure>

            <figure>
              <figcaption>Grad-CAM overlay</figcaption>
              <div className="image-frame">
                {gradCam ? (
                  <img
                    src={`data:image/png;base64,${gradCam.overlay_base64}`}
                    alt={`Grad-CAM overlay for ${gradCam.target_label}`}
                  />
                ) : (
                  <p className="image-placeholder">
                    Run classification, then select Explain beside a label.
                  </p>
                )}
              </div>
              <p className="image-caption">
                {gradCam
                  ? `Target: ${displayLabel(gradCam.target_label)}`
                  : "No explanation selected"}
              </p>
            </figure>
          </div>

          {gradCam && (
            <div className="attribution-summary">
              <h3>Grad-CAM: {displayLabel(gradCam.target_label)}</h3>
              <p>
                Target model score:{" "}
                {(gradCam.target_probability * 100).toFixed(2)}%
              </p>
              <p className="muted">{gradCam.limitation}</p>
            </div>
          )}

          <p className="footnote">
            Attribution visualizes model sensitivity. It does not confirm a
            finding or establish that the model used clinically relevant regions.
          </p>
        </section>

        <section className="panel results-panel" aria-labelledby="results-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">03 / MODEL OUTPUT</p>
              <h2 id="results-heading">Model output</h2>
            </div>
            {result && (
              <span className="badge">{result.predictions.length} labels</span>
            )}
          </div>

          {result ? (
            <>
              <div className="model-meta">
                <span>Model: {result.model}</span>
                <code>{result.model_artifact}</code>
              </div>

              <p className="muted score-note">
                Scores sorted highest first. Scores are not a diagnosis or a
                validated estimate of individual disease risk.
              </p>

              <ul className="prediction-list" aria-label="Classification scores">
                {predictions.map((prediction) => (
                  <li
                    key={prediction.label}
                    className={
                      gradCam?.target_label === prediction.label
                        ? "prediction-row selected"
                        : "prediction-row"
                    }
                  >
                    <div className="prediction-content">
                      <span className="prediction-label">
                        {displayLabel(prediction.label)}:{" "}
                        <span className="score">
                          {(prediction.probability * 100).toFixed(2)}%
                        </span>
                      </span>
                      <div className="score-track" aria-hidden="true">
                        <div
                          className="score-fill"
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(0, prediction.probability * 100),
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                    <button
                      className="button-small"
                      type="button"
                      aria-label={`Explain ${displayLabel(prediction.label)}`}
                      onClick={() => handleGradCam(prediction.label)}
                      disabled={busy}
                    >
                      Explain
                    </button>
                  </li>
                ))}
              </ul>

              <p className="footnote">{result.limitation}</p>

              <div className="review-panel">
                <h3>Reviewer feedback</h3>
                <p className="muted">
                  Record a review of this output. This is not clinical approval
                  and does not retrain the model.
                </p>
                <div className="review-actions">
                  <button
                    type="button"
                    onClick={() => handleReview("approve")}
                    disabled={busy}
                  >
                    Approve review
                  </button>
                  <button
                    className="button-reject"
                    type="button"
                    onClick={() => handleReview("reject")}
                    disabled={busy}
                  >
                    Reject review
                  </button>
                </div>
                {reviewNotice && (
                  <p className="review-confirmation" role="alert">
                    {reviewNotice}
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <h3>No classification yet</h3>
              <p>
                Select a chest X-ray and run classification. Label scores and
                explanation controls will appear here.
              </p>
            </div>
          )}
        </section>
      </div>

      <footer>
        Educational research interface · Outputs require independent evaluation.
      </footer>
    </main>
  );
}