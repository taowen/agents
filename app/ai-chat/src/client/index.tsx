import "./styles.css";
import { useState, useRef, useEffect } from "react";
import * as Sentry from "@sentry/react";
import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
  useRouteError,
  useLocation,
  useNavigationType,
  createRoutesFromChildren,
  matchRoutes
} from "react-router";
import { ThemeProvider } from "../lib/agents-ui/hooks";
import { useSessions } from "./api";
import { RootLayout } from "./RootLayout";
import { AuthLayout } from "./AuthLayout";
import { Chat } from "./Chat";
import { DevicePage } from "./DevicePage";
import { SettingsPage } from "./SettingsPage";
import { MemoryPage } from "./MemoryPage";
import { UsagePage } from "./UsagePage";

function IndexRedirect() {
  const { sessions, isLoading, createSession } = useSessions();
  const [createdId, setCreatedId] = useState<string | null>(null);
  const creating = useRef(false);

  if (
    !isLoading &&
    sessions &&
    sessions.length === 0 &&
    !creating.current &&
    !createdId
  ) {
    creating.current = true;
    createSession().then((s) => setCreatedId(s.id));
  }

  const targetId = createdId ?? sessions?.[0]?.id;
  if (targetId) {
    return <Navigate to={`/s/${targetId}`} replace />;
  }

  return null;
}

function RouteErrorFallback() {
  const error = useRouteError();
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);
  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <h2>Something went wrong</h2>
      <p style={{ color: "#888" }}>
        {error instanceof Error ? error.message : "Unknown error"}
      </p>
      <a href="/" style={{ color: "#3b82f6" }}>
        Go home
      </a>
    </div>
  );
}

declare const __SENTRY_DSN__: string;

if (__SENTRY_DSN__) {
  Sentry.init({
    dsn: __SENTRY_DSN__,
    integrations: [
      Sentry.reactRouterV7BrowserTracingIntegration({
        useEffect,
        useLocation,
        useNavigationType,
        createRoutesFromChildren,
        matchRoutes
      })
    ],
    tracesSampleRate: 1.0
  });
}

const sentryCreateBrowserRouter =
  Sentry.wrapCreateBrowserRouterV7(createBrowserRouter);

const router = sentryCreateBrowserRouter([
  {
    path: "/device",
    element: <DevicePage />,
    errorElement: <RouteErrorFallback />
  },
  {
    element: <RootLayout />,
    errorElement: <RouteErrorFallback />,
    children: [
      {
        element: <AuthLayout />,
        children: [
          {
            index: true,
            element: <IndexRedirect />
          },
          {
            path: "s/:sessionId",
            element: <Chat />
          },
          { path: "settings", element: <SettingsPage /> },
          { path: "memory", element: <MemoryPage /> },
          { path: "usage", element: <UsagePage /> }
        ]
      }
    ]
  }
]);

const root = createRoot(document.getElementById("root")!);
root.render(
  <Sentry.ErrorBoundary fallback={<p>Something went wrong.</p>}>
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  </Sentry.ErrorBoundary>
);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js");
}
