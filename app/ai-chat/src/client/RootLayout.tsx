import { Outlet } from "react-router";
import { LoginPage } from "./LoginPage";
import { useAuth } from "./api";
import { AppShellSkeleton } from "./Skeleton";

export function RootLayout() {
  const { user, authenticated, isLoading } = useAuth();

  if (isLoading) {
    return <AppShellSkeleton />;
  }

  if (!authenticated) {
    return <LoginPage />;
  }

  return <Outlet context={{ user: user! }} />;
}
