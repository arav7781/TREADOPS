import type { Role } from "./types";

/** Mirrors treadops_api.security.Principal.has: admin implies everything; any role implies viewer. */
export function hasRole(roles: readonly string[] | undefined, role: Role): boolean {
  if (!roles?.length) return false;
  if (roles.includes("admin")) return true;
  if (role === "viewer") return true;
  return roles.includes(role);
}
