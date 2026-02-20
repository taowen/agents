# Android Browser Agent

A WebView-based browser with an LLM agent that can automate web tasks (click, type, navigate, screenshot).

## Architecture

```
┌─────────────────────────────────────────┐
│  AgentRunner (background Java thread)   │
│  - LLM loop: system prompt → tool calls │
│  - HTTP calls via HttpURLConnection     │
│  - Calls AgentBridge methods directly   │
└────────────────┬────────────────────────┘
                 │ plain Java calls
┌────────────────▼────────────────────────┐
│  AgentBridge (utility class)            │
│  - evaluateJavascript() on UI thread    │
│  - CountDownLatch to block until done   │
│  - DOM extraction, click, type, scroll  │
└────────────────┬────────────────────────┘
                 │ runOnUiThread
┌────────────────▼────────────────────────┐
│  Visible WebView (the actual browser)   │
│  - dom-extractor.js injected on demand  │
│  - Single WebView, no contention        │
└─────────────────────────────────────────┘
```

Key design decision: the agent loop runs in a **plain Java thread**, not a hidden WebView. An earlier design used a hidden WebView (JS agent loop) + visible WebView (browser), but this caused a **deadlock** — both WebViews share one renderer process and one JS thread. When the hidden WebView's JS called a `@JavascriptInterface` method that then called `evaluateJavascript()` on the visible WebView, the visible WebView's JS couldn't execute because the shared JS thread was blocked. Moving the agent loop to Java eliminates this entirely.

## Build

```bash
# Build dom-extractor.js (the only TS→JS asset needed)
npm run build

# Build APK
cd app && ../gradlew assembleDebug && cd ..
# or from project root:
./gradlew assembleDebug
```

## Install & Launch

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.androidbrowser/.MainActivity
```

## Configure LLM

```bash
adb shell "am start -n com.example.androidbrowser/.MainActivity \
  --es configure_llm true \
  --es llm_base_url 'https://api.openai.com/v1' \
  --es llm_api_key 'sk-...' \
  --es llm_model 'gpt-4o'"
```

Config is persisted in SharedPreferences (`agent_prefs`).

## Run a Task

Via broadcast (works even if app is already open):

```bash
adb shell "am broadcast -n com.example.androidbrowser/.TaskReceiver \
  --es task 'search for weather in Beijing on baidu.com'"
```

Or type in the task input field in the app and tap "Run".

## View Logs

```bash
adb logcat -s AgentRunner AgentBridge
```

**Tip**: logcat buffer rotates fast during agent runs. Redirect to a file for complete logs:
```bash
adb logcat -s AgentRunner AgentBridge > /tmp/agent-test.log &
# ... run task, then kill the logcat process
```

## Performance Tuning Log

### v2 — Page load detection + prompt improvements (2025-02-20)

**Test task**: "search for weather in Beijing on baidu.com"

| Metric | v1 (hardcoded sleeps) | v2 (page load detection) |
|---|---|---|
| Steps | 9 | 6 |
| Total time | ~22s | ~23s (LLM-dominated) |
| Action overhead | ~5.5s (sleeps) | ~0.9s |
| LLM round-trips saved | — | 3 |

Step-by-step trace (v2):
```
Step 1: goto_url(baidu.com)  — 465ms (was 1500ms sleep)
Step 2: get_page              — 12ms
Step 3: type(12, "北京天气")   — 58ms (type now works on Google/Baidu textarea)
Step 4: click(13, search btn) — 348ms (was 500ms sleep)
Step 5: get_page              — 16ms
Step 6: DONE                  — LLM summarizes results
```

Changes that enabled this:

1. **Page load detection** (`AgentBridge.waitForPageLoad`) — `CountDownLatch` wired to `WebViewClient.onPageStarted/onPageFinished`. Replaces hardcoded `Thread.sleep()`. For `goto_url`: pre-arms latch before `loadUrl()`. For `click`/`go_back`: polls up to 300ms for `onPageStarted`, then waits on latch.

2. **Removed `wait` tool** — LLM was calling `wait(2000)` to guess delays. Now `click`/`goto_url`/`go_back` auto-wait, so the tool is unnecessary.

3. **Prompt: "never call get_page twice in a row"** — Eliminated a redundant LLM round-trip (previously steps 6 & 7 were both get_page).

4. **React-compatible `typeText()`** — Uses `Object.getOwnPropertyDescriptor(HTMLInputElement/HTMLTextAreaElement.prototype, 'value').set` + `InputEvent`. Fixed: element tagName check to pick correct prototype (was always using HTMLInputElement, broke on `<textarea>`).

5. **Full click event sequence** — `pointerdown → mousedown → pointerup → mouseup → click` with `getBoundingClientRect()` coordinates. Fixes React/Vue sites that listen for pointer events.

6. **DOM `innerText` fallback** — `describeElement()` now falls back to `el.innerText` when `getDirectText()` returns empty (catches `<button><span>Search</span></button>`).

7. **Screenshot quality** — JPEG 50% → 70%.

**Gotchas found during testing**:
- `adb shell am start --es key value` truncates at the first space. Use: `adb shell "am start ... --es key 'multi word value'"` (outer double quotes, inner single quotes).
- `onPageStarted()` must NOT reset the `CountDownLatch` if already loading (pre-armed by `navigateTo`), otherwise `waitForPageLoad()` waits on a stale latch reference and times out.
- `Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')` always returns a descriptor even if the element is a `<textarea>` — calling `.set.call(textarea, ...)` throws TypeError. Must check `el.tagName` first.

## Files

| File | Purpose |
|------|---------|
| `AgentRunner.java` | Agent loop (Runnable on background thread). LLM calls, tool dispatch, message trimming. |
| `AgentBridge.java` | Utility class bridging agent thread → visible WebView. DOM extraction, click, type, scroll, screenshot. |
| `MainActivity.java` | Activity with WebView browser + agent UI. Starts AgentRunner on `new Thread()`. |
| `TaskReceiver.java` | BroadcastReceiver to trigger agent via `adb shell am broadcast`. |
| `src/dom-extractor.ts` | Injected into visible WebView to extract DOM tree with interactive element IDs. |
| `src/prompt.ts` | (Reference only) System prompt & tool definitions — now hardcoded in AgentRunner.java. |
