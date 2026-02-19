import { tool } from "ai";
import { z } from "zod";
import puppeteer from "@cloudflare/puppeteer";
import type { Browser, Page } from "@cloudflare/puppeteer";

export interface BrowserState {
  browser: Browser | null;
  page: Page | null;
  closeTimeout: ReturnType<typeof setTimeout> | null;
}

export function createBrowserState(): BrowserState {
  return { browser: null, page: null, closeTimeout: null };
}

export async function getOrLaunchPage(
  state: BrowserState,
  browserBinding: Fetcher
): Promise<Page> {
  // Reset idle timer
  if (state.closeTimeout) {
    clearTimeout(state.closeTimeout);
    state.closeTimeout = null;
  }
  state.closeTimeout = setTimeout(() => closeBrowser(state), 5 * 60 * 1000);

  // Reuse existing if still valid
  if (state.browser?.isConnected() && state.page && !state.page.isClosed()) {
    return state.page;
  }

  // Clean up stale references
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
  return state.page;
}

export async function closeBrowser(state: BrowserState): Promise<void> {
  if (state.closeTimeout) {
    clearTimeout(state.closeTimeout);
    state.closeTimeout = null;
  }
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

function toModelOutput({
  output
}: {
  toolCallId: string;
  input: unknown;
  output: unknown;
}) {
  const result = output as {
    action: string;
    success: boolean;
    url: string;
    title: string;
    text: string;
    screenshot: string;
    error?: string;
  };
  const parts: Array<
    | { type: "text"; text: string }
    | { type: "media"; data: string; mediaType: string }
  > = [];
  // Text summary
  const lines: string[] = [];
  lines.push(`Action: ${result.action} | Success: ${result.success}`);
  if (result.error) lines.push(`Error: ${result.error}`);
  if (result.url) lines.push(`URL: ${result.url}`);
  if (result.title) lines.push(`Title: ${result.title}`);
  if (result.text) lines.push(`Text:\n${result.text}`);
  parts.push({ type: "text", text: lines.join("\n") });
  // Screenshot as image
  if (result.screenshot) {
    parts.push({
      type: "media",
      data: result.screenshot,
      mediaType: "image/png"
    });
  }
  return { type: "content" as const, value: parts };
}

export function createBrowserTool(
  state: BrowserState,
  browserBinding: Fetcher
) {
  return tool({
    description:
      "Browse web pages using a real browser. Supports navigation, clicking, typing, screenshots, scrolling, and text extraction. " +
      "Use this for JavaScript-heavy pages, SPAs, or when you need to interact with a page. " +
      "Each action returns a screenshot of the current page state.",
    inputSchema: z.object({
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
    execute: async ({ action, url, selector, text, direction, cookies }) => {
      try {
        if (action === "close") {
          await closeBrowser(state);
          return {
            action,
            success: true,
            url: "",
            title: "",
            text: "Browser closed",
            screenshot: ""
          };
        }

        const page = await getOrLaunchPage(state, browserBinding);

        switch (action) {
          case "goto": {
            if (!url)
              return {
                action,
                success: false,
                error: "url is required for goto action",
                url: "",
                title: "",
                text: "",
                screenshot: ""
              };
            await page.goto(url, {
              waitUntil: "networkidle0",
              timeout: 30000
            });
            const s = await capturePageState(page);
            return { action, success: true, ...s };
          }
          case "click": {
            if (!selector)
              return {
                action,
                success: false,
                error: "selector is required for click action",
                url: "",
                title: "",
                text: "",
                screenshot: ""
              };
            await page.waitForSelector(selector, { timeout: 5000 });
            await page.click(selector);
            try {
              await page.waitForNetworkIdle({ timeout: 5000 });
            } catch (err) {
              console.warn(
                "Browser: waitForNetworkIdle timed out after click:",
                err
              );
            }
            const s = await capturePageState(page);
            return { action, success: true, ...s };
          }
          case "type": {
            if (!selector)
              return {
                action,
                success: false,
                error: "selector is required for type action",
                url: "",
                title: "",
                text: "",
                screenshot: ""
              };
            if (!text)
              return {
                action,
                success: false,
                error: "text is required for type action",
                url: "",
                title: "",
                text: "",
                screenshot: ""
              };
            await page.waitForSelector(selector, { timeout: 5000 });
            await page.click(selector, { clickCount: 3 });
            await page.type(selector, text);
            const s = await capturePageState(page);
            return { action, success: true, ...s };
          }
          case "screenshot": {
            const s = await capturePageState(page);
            return { action, success: true, ...s };
          }
          case "scroll": {
            const scrollDir = direction === "up" ? -500 : 500;
            await page.evaluate((d) => window.scrollBy(0, d), scrollDir);
            await new Promise((r) => setTimeout(r, 500));
            const s = await capturePageState(page);
            return { action, success: true, ...s };
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
            return { action, success: true, ...s, text: extracted };
          }
          case "set_cookies": {
            if (!cookies)
              return {
                action,
                success: false,
                error: "cookies is required for set_cookies action",
                url: "",
                title: "",
                text: "",
                screenshot: ""
              };
            const parsed = JSON.parse(cookies);
            const cookieArray = Array.isArray(parsed) ? parsed : [parsed];
            await page.setCookie(...cookieArray);
            const s = await capturePageState(page);
            return {
              action,
              success: true,
              ...s,
              text: `Set ${cookieArray.length} cookie(s)`
            };
          }
          default:
            return {
              action,
              success: false,
              error: `Unknown action: ${action}`,
              url: "",
              title: "",
              text: "",
              screenshot: ""
            };
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        console.error(`Browser: action "${action}" failed:`, errorMsg);
        // Try to capture current state even on error
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
        } catch (captureErr) {
          console.error("Browser: failed to capture error state:", captureErr);
        }
        return {
          action,
          success: false,
          error: errorMsg,
          url: pageUrl,
          title: pageTitle,
          text: "",
          screenshot
        };
      }
    },
    toModelOutput
  });
}
