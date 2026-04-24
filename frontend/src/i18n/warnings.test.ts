import { describe, expect, it } from "vitest";
import { translateWarning } from "./warnings";

describe("translateWarning", () => {
  it("returns russian spec for known code", () => {
    const spec = translateWarning("hmm.guards_failed");
    expect(spec?.short).toMatch(/HMM/);
    expect(spec?.tone).toBe("warning");
  });

  it("returns null for unknown code (caller falls back to backend message)", () => {
    expect(translateWarning("unknown.code")).toBeNull();
  });
});
