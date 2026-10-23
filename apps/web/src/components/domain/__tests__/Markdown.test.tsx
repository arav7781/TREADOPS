import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Markdown } from "@/components/domain/Markdown";

describe("Markdown", () => {
  it("renders headings, rules and numbered lists instead of showing the markup", () => {
    const { container } = render(<Markdown text={[
      "## Verdict: Damaged — Defect flagged",
      "",
      "Evidence:",
      "",
      "- defective **0.916**",
      "- quality accepted",
      "",
      "---",
      "",
      "### Recommended next steps",
      "",
      "1. Remove from intake",
      "2. Review the heatmap",
    ].join("\n")} />);
    expect(screen.getByRole("heading", { name: "Verdict: Damaged — Defect flagged" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Recommended next steps" })).toBeInTheDocument();
    expect(container.querySelector("hr")).not.toBeNull();
    expect(container.querySelectorAll("ol > li")).toHaveLength(2);
    expect(container.querySelectorAll("ul > li")).toHaveLength(2);
    expect(container.textContent).not.toMatch(/##|---|\*\*/);
  });

  it("renders tables and knowledge-base citation chips", () => {
    render(<Markdown text={"| class | p |\n|---|---|\n| defective | 0.916 |\n\nSee [KB:kb-004]."} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "0.916" })).toBeInTheDocument();
    expect(screen.getByTitle("Knowledge-base citation")).toHaveTextContent("KB:kb-004");
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("never renders raw HTML or images from an answer", () => {
    const { container } = render(<Markdown text={'<img src="https://evil.example/x.png"> hi ![x](https://evil.example/y.png) [ok](https://example.com)'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByRole("link", { name: "ok" })).toHaveAttribute("rel", "noopener noreferrer nofollow");
  });
});
