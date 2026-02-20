import { z, type AgentTool, type AgentToolResult } from "pi";
import puppeteer from "@cloudflare/puppeteer";
import type { Browser, Page } from "@cloudflare/puppeteer";
import * as Sentry from "@sentry/cloudflare";

export interface BrowserState {
  browser: Browser | null;
  page: Page | null;
}

export function createBrowserState(): BrowserState {
  return { browser: null, page: null };
}

export async function getOrLaunchPage(
  state: BrowserState,
  browserBinding: Fetcher
): Promise<Page> {
  if (state.browser?.isConnected() && state.page && !state.page.isClosed()) {
    return state.page;
  }

  await closeBrowser(state);

  console.log("Browser: launching new instance");
  state.browser = await puppeteer.launch(browserBinding, {
    keep_alive: 600000
  });
  state.page = await state.browser.newPage();
  await state.page.setUserAgent(
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
  );
  await state.page.setViewport({ width: 1280, height: 720 });
  Sentry.addBreadcrumb({
    category: "browser",
    message: "Browser launched",
    level: "info"
  });
  return state.page;
}

export async function closeBrowser(state: BrowserState): Promise<void> {
  const hadBrowser = state.browser !== null;
  try {
    if (state.page && !state.page.isClosed()) await state.page.close();
  } catch (err) {
    console.error("Browser: error closing page:", err);
  }
  try {
    if (state.browser?.isConnected()) await state.browser.close();
  } catch (err) {
    console.error("Browser: error closing browser:", err);
  }
  state.page = null;
  state.browser = null;
  if (hadBrowser) {
    Sentry.addBreadcrumb({
      category: "browser",
      message: "Browser closed",
      level: "info"
    });
  }
  console.log("Browser: closed");
}

async function capturePageState(page: Page): Promise<{
  screenshot: string;
  url: string;
  title: string;
  text: string;
}> {
  const screenshot = (await page.screenshot({
    encoding: "base64"
  })) as string;
  const title = await page.title();
  const url = page.url();
  const text = await page.evaluate(() => {
    const body = document.body;
    if (!body) return "";
    return body.innerText.slice(0, 8000);
  });
  return { screenshot, url, title, text };
}

interface BrowserResult {
  action: string;
  success: boolean;
  url: string;
  title: string;
  text: string;
  screenshot: string;
  error?: string;
}

function resultToToolOutput(
  result: BrowserResult
): AgentToolResult<BrowserResult> {
  const lines: string[] = [];
  lines.push(`Action: ${result.action} | Success: ${result.success}`);
  if (result.error) lines.push(`Error: ${result.error}`);
  if (result.url) lines.push(`URL: ${result.url}`);
  if (result.title) lines.push(`Title: ${result.title}`);
  if (result.text) lines.push(`Text:\n${result.text}`);

  const content: AgentToolResult<BrowserResult>["content"] = [
    { type: "text", text: lines.join("\n") }
  ];

  if (result.screenshot) {
    content.push({
      type: "image",
      data: result.screenshot,
      mimeType: "image/png"
    });
  }

  return { content, details: result };
}

export function createBrowserTool(
  state: BrowserState,
  browserBinding: Fetcher
): AgentTool {
  return {
    name: "browser",
    description:
      "Browse web pages using a real browser. Supports navigation, clicking, typing, screenshots, scrolling, and text extraction. " +
      "Use this for JavaScript-heavy pages, SPAs, or when you need to interact with a page. " +
      "Each action returns a screenshot of the current page state.",
    label: "Browser",
    parameters: z.object({
      action: z
        .enum([
          "goto",
          "click",
          "type",
          "screenshot",
          "scroll",
          "extract",
          "set_cookies",
          "close"
        ])
        .describe("The browser action to perform"),
      url: z
        .string()
        .optional()
        .describe("URL to navigate to (for goto action)"),
      selector: z
        .string()
        .optional()
        .describe(
          "CSS selector for the target element (for click, type, extract actions)"
        ),
      text: z.string().optional().describe("Text to type (for type action)"),
      direction: z
        .enum(["up", "down"])
        .optional()
        .describe("Scroll direction (for scroll action)"),
      cookies: z
        .string()
        .optional()
        .describe(
          "JSON array of cookie objects for set_cookies action. Each cookie needs: name, value, domain. Optional: path, expires, httpOnly, secure, sameSite."
        )
    }),
    execute: async (
      _toolCallId,
      params
    ): Promise<AgentToolResult<BrowserResult>> => {
      const { action, url, selector, text, direction, cookies } = params as {
        action: string;
        url?: string;
        selector?: string;
        text?: string;
        direction?: "up" | "down";
        cookies?: string;
      };

      return Sentry.startSpan(
        { name: `browser.${action}`, op: "tool.browser" },
        async () => {
          Sentry.addBreadcrumb({
            category: "browser",
            message: `browser.${action}`,
            data: { url, selector, text: text?.slice(0, 100) },
            level: "info"
          });

          try {
            if (action === "close") {
              await closeBrowser(state);
              return resultToToolOutput({
                action,
                success: true,
                url: "",
                title: "",
                text: "Browser closed",
                screenshot: ""
              });
            }

            const page = await getOrLaunchPage(state, browserBinding);

            switch (action) {
              case "goto": {
                if (!url) {
                  return resultToToolOutput({
                    action,
                    success: false,
                    error: "url is required for goto action",
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  });
                }
                await page.goto(url, {
                  waitUntil: "networkidle0",
                  timeout: 30000
                });
                const s = await capturePageState(page);
                return resultToToolOutput({ action, success: true, ...s });
              }
              case "click": {
                if (!selector) {
                  return resultToToolOutput({
                    action,
                    success: false,
                    error: "selector is required for click action",
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  });
                }
                await page.waitForSelector(selector, { timeout: 5000 });
                await page.click(selector);
                try {
                  await page.waitForNetworkIdle({ timeout: 5000 });
                } catch {
                  console.warn(
                    "Browser: waitForNetworkIdle timed out after click"
                  );
                }
                const s = await capturePageState(page);
                return resultToToolOutput({ action, success: true, ...s });
              }
              case "type": {
                if (!selector) {
                  return resultToToolOutput({
                    action,
                    success: false,
                    error: "selector is required for type action",
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  });
                }
                if (!text) {
                  return resultToToolOutput({
                    action,
                    success: false,
                    error: "text is required for type action",
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  });
                }
                await page.waitForSelector(selector, { timeout: 5000 });
                await page.click(selector, { clickCount: 3 });
                await page.type(selector, text);
                const s = await capturePageState(page);
                return resultToToolOutput({ action, success: true, ...s });
              }
              case "screenshot": {
                const s = await capturePageState(page);
                return resultToToolOutput({ action, success: true, ...s });
              }
              case "scroll": {
                const scrollDir = direction === "up" ? -500 : 500;
                await page.evaluate((d) => window.scrollBy(0, d), scrollDir);
                await new Promise((r) => setTimeout(r, 500));
                const s = await capturePageState(page);
                return resultToToolOutput({ action, success: true, ...s });
              }
              case "extract": {
                let extracted: string;
                if (selector) {
                  extracted = await page.$eval(selector, (el) =>
                    (el as HTMLElement).innerText.slice(0, 16000)
                  );
                } else {
                  extracted = await page.evaluate(() =>
                    document.body.innerText.slice(0, 16000)
                  );
                }
                const s = await capturePageState(page);
                return resultToToolOutput({
                  action,
                  success: true,
                  ...s,
                  text: extracted
                });
              }
              case "set_cookies": {
                if (!cookies) {
                  return resultToToolOutput({
                    action,
                    success: false,
                    error: "cookies is required for set_cookies action",
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  });
                }
                const parsed = JSON.parse(cookies);
                const cookieArray = Array.isArray(parsed) ? parsed : [parsed];
                await page.setCookie(...cookieArray);
                const s = await capturePageState(page);
                return resultToToolOutput({
                  action,
                  success: true,
                  ...s,
                  text: `Set ${cookieArray.length} cookie(s)`
                });
              }
              default:
                return resultToToolOutput({
                  action,
                  success: false,
                  error: `Unknown action: ${action}`,
                  url: "",
                  title: "",
                  text: "",
                  screenshot: ""
                });
            }
          } catch (err) {
            Sentry.captureException(err);
            const errorMsg = err instanceof Error ? err.message : String(err);
            console.error(`Browser: action "${action}" failed:`, errorMsg);
            let screenshot = "";
            let pageUrl = "";
            let pageTitle = "";
            try {
              if (state.page && !state.page.isClosed()) {
                const s = await capturePageState(state.page);
                screenshot = s.screenshot;
                pageUrl = s.url;
                pageTitle = s.title;
              }
            } catch {
              console.error("Browser: failed to capture error state");
            }
            return resultToToolOutput({
              action,
              success: false,
              error: errorMsg,
              url: pageUrl,
              title: pageTitle,
              text: "",
              screenshot
            });
          }
        }
      );
    }
  };
}
