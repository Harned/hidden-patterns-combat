import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("renders baseline label in Russian", () => {
    render(<StatusBadge status="baseline_only" />);
    expect(screen.getByText("Baseline")).toBeInTheDocument();
  });

  it("renders needs_column_mapping label", () => {
    render(<StatusBadge status="needs_column_mapping" />);
    expect(screen.getByText("Нужен column mapping")).toBeInTheDocument();
  });

  it("renders hmm_ready label", () => {
    render(<StatusBadge status="hmm_ready" />);
    expect(screen.getByText("HMM готов")).toBeInTheDocument();
  });
});
