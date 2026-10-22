import type { Role } from "./types";

/** Mirrors treadops_api.security.Principal.has: admin implies everything; any role implies viewer. */
export function hasRole(roles: readonly string[] | undefined, role: Role): boolean {
  if (!roles?.length) return false;
  if (roles.includes("admin")) return true;
  if (role === "viewer") return true;
  return roles.includes(role);
}

export function primaryRole(roles: readonly string[] | undefined): string {
  if (!roles?.length) return "signed out";
  for (const r of ["admin", "ml_engineer", "inspector", "viewer"]) if (roles.includes(r)) return r.replace("_", " ");
  return roles[0];
}
