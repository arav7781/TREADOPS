import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ErrorState } from "@/components/ui";
import { ApiError } from "@/lib/api";

describe("ErrorState", () => {
  it("shows code, status and request id with a retry path", async () => {
    const retry = vi.fn();
    render(<ErrorState error={new ApiError(503, "model_unavailable", "No trained model available.", "req-42", true)} onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("No trained model available.");
    expect(screen.getByText(/model_unavailable · HTTP 503 · request req-42/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(retry).toHaveBeenCalledOnce();
  });
  it("renders 403 as an access-denied state", () => {
    render(<ErrorState error={new ApiError(403, "forbidden", "This action requires the admin role.")} />);
    expect(screen.getByText("Access denied")).toBeInTheDocument();
    expect(screen.getByText("This action requires the admin role.")).toBeInTheDocument();
  });
});
