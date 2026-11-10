import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import App from "./App";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

test("renders disclaimer and handles classification success", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({
      model: "densenet121",
      model_artifact: "best_classifier.pt",
      threshold: 0.5,
      predictions: [
        {
          label: "Cardiomegaly",
          probability: 0.6,
          positive_at_threshold: true,
        },
      ],
      limitation: "Educational model output only.",
    }),
  });

  render(<App />);

  expect(screen.getByText(/not for clinical use/i)).toBeInTheDocument();

  const input = screen.getByLabelText("Chest X-ray image");
  const file = new File(["image"], "xray.png", { type: "image/png" });

  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await waitFor(() => {
    expect(screen.getByRole("status")).toHaveTextContent(
      "Classification completed.",
    );
  });

  const scores = screen.getByRole("list", {
    name: "Classification scores",
  });
  const rows = within(scores).getAllByRole("listitem");

  expect(rows).toHaveLength(1);
  expect(rows[0]).toHaveTextContent("Cardiomegaly: 60.00%");
  expect(
    within(rows[0]).getByRole("button", {
      name: "Explain Cardiomegaly",
    }),
  ).toBeEnabled();
  expect(screen.getByText("best_classifier.pt")).toBeInTheDocument();
});

test("shows backend errors", async () => {
  fetchMock.mockResolvedValue({
    ok: false,
    json: async () => ({
      detail: { message: "Model unavailable." },
    }),
  });

  render(<App />);

  fireEvent.change(screen.getByLabelText("Chest X-ray image"), {
    target: {
      files: [
        new File(["image"], "xray.png", {
          type: "image/png",
        }),
      ],
    },
  });

  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await waitFor(() => {
    expect(screen.getByRole("status")).toHaveTextContent("Model unavailable.");
  });

  expect(
    screen.queryByRole("list", { name: "Classification scores" }),
  ).not.toBeInTheDocument();

  expect(
    screen.getByRole("button", { name: "Run classification" }),
  ).toBeEnabled();
});

test("clears previous classification when a different image is selected", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({
      model: "densenet121",
      model_artifact: "best_classifier.pt",
      threshold: 0.5,
      predictions: [
        {
          label: "Cardiomegaly",
          probability: 0.6,
          positive_at_threshold: true,
        },
      ],
      limitation: "Educational model output only.",
    }),
  });

  render(<App />);

  const input = screen.getByLabelText("Chest X-ray image");

  fireEvent.change(input, {
    target: {
      files: [new File(["first"], "first.png", { type: "image/png" })],
    },
  });

  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await screen.findByRole("list", { name: "Classification scores" });

  fireEvent.change(input, {
    target: {
      files: [new File(["second"], "second.png", { type: "image/png" })],
    },
  });

  expect(
    screen.queryByRole("list", { name: "Classification scores" }),
  ).not.toBeInTheDocument();

  expect(screen.getByText("No classification yet")).toBeInTheDocument();

  expect(screen.getByRole("status")).toHaveTextContent(
    "Image selected. Run classification to view model scores.",
  );
});

test("shows an explicit confirmation after approving a review", async () => {
  fetchMock
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        model: "densenet121",
        model_artifact: "best_classifier.pt",
        threshold: 0.5,
        predictions: [
          {
            label: "Cardiomegaly",
            probability: 0.6,
            positive_at_threshold: true,
          },
        ],
        limitation: "Educational model output only.",
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: "review-1",
        reference: "xray.png",
        action: "approve",
        model_artifact: "best_classifier.pt",
        created_at: "2026-11-08T00:00:00Z",
      }),
    });

  render(<App />);

  fireEvent.change(screen.getByLabelText("Chest X-ray image"), {
    target: {
      files: [new File(["image"], "xray.png", { type: "image/png" })],
    },
  });
  fireEvent.click(screen.getByRole("button", { name: "Run classification" }));

  await screen.findByRole("list", { name: "Classification scores" });
  fireEvent.click(screen.getByRole("button", { name: "Approve review" }));

  const confirmation = await screen.findByRole("alert");
  expect(confirmation).toHaveTextContent("Review approved successfully.");
  expect(fetchMock).toHaveBeenLastCalledWith(
    "http://localhost:8000/reviews",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        reference: "xray.png",
        action: "approve",
        model_artifact: "best_classifier.pt",
      }),
    }),
  );
});