import "./styles.css";
import * as Sentry from "@sentry/react";
import { createRoot } from "react-dom/client";
import { ThemeProvider } from "@cloudflare/agents-ui/hooks";
import App from "./App";

declare const __SENTRY_DSN__: string;

if (__SENTRY_DSN__) {
  Sentry.init({
    dsn: __SENTRY_DSN__,
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: 1.0
  });
}

const root = createRoot(document.getElementById("root")!);
root.render(
  <Sentry.ErrorBoundary fallback={<p>Something went wrong.</p>}>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </Sentry.ErrorBoundary>
);
