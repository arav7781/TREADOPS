import { AppShell } from "@/components/AppShell";

export default function AuthedLayout({ children }: LayoutProps<"/">) {
  return <AppShell>{children}</AppShell>;
}
