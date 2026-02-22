# RN Agent

Android Accessibility Service agent powered by React Native + standalone Hermes.

LLM generates JavaScript code, which is `eval()`'d in a standalone Hermes runtime inside the AccessibilityService process. The JS code calls synchronous C++ host functions to interact with the accessibility tree.

## Architecture

```
agent-standalone.ts  ──>  runAgent(task, configJson)
                           │
                           ├── llm-client.ts: callLLM()  ──>  http_post() [sync C++ host fn]
                           │
                           ├── conversation-manager.ts: trimMessages() / compactHistory()
                           │
                           └── executeCode(code)  [eval() in Hermes]
                                 │
                                 ├── get_screen()        ──>  accessibility tree as text
                                 ├── take_screenshot()   ──>  base64 JPEG (vision)
                                 ├── click(target)       ──>  by text / desc / coords
                                 ├── long_click(target)  ──>  same syntax as click
                                 ├── scroll(dir)         ──>  screen scroll
                                 ├── scroll_element(text, dir)
                                 ├── type_text(text)     ──>  type into focused input
                                 ├── press_home / press_back / press_recents
                                 ├── show_notifications / launch_app / list_apps
                                 ├── sleep(ms)
                                 └── log(msg)
```

Key files:

| File | Role |
|------|------|
| `src/agent-standalone.ts` | Agent loop entry point (runs in standalone Hermes) — tool dispatch, code execution |
| `src/llm-client.ts` | LLM API client with retry logic (`callLLM`, `http_post` host fn) |
| `src/conversation-manager.ts` | Conversation history trimming, compaction (summarization), state persistence |
| `src/host-api.ts` | Host function definitions and metadata |
| `src/prompt.ts` | System prompt and tool definitions for the LLM |
| `src/App.tsx` | RN UI: device login, task input, log viewer, WebSocket cloud connection |
| `src/NativeAccessibilityBridge.ts` | TurboModule spec for JS↔Java bridge |
| `src/types.ts` | Shared TypeScript types |
| `AccessibilityBridgeModule.java` | Sync native methods bridging to SelectToSpeakService |
| `SelectToSpeakService.java` | Core accessibility service (reads UI tree, performs clicks/scrolls, takes screenshots) |
| `HermesAgentRunner.java` | Standalone Hermes runtime that loads `agent-standalone.js` |
| `TaskReceiver.java` | BroadcastReceiver that emits `onTaskReceived` event to RN |

## Build & Deploy

JS is bundled into the APK (no Metro dev server needed — `debuggableVariants = []` in build.gradle).

### Debug build (local testing)

```sh
cd app/rn/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

### Publish release APK

Builds a release APK, uploads it to R2, and updates the download link in `app/ai-chat`:

```sh
npm run publish:apk -w app/rn
# Then deploy ai-chat for the download page to serve the new APK:
npm run deploy -w app/ai-chat
```

The APK filename includes the git commit hash (e.g. `connect-screen-a1b2c3d.apk`). The download page is at `https://ai.connect-screen.com/download`.

## LLM Config & Device Login

Config is obtained via device auth flow — no local config file needed:

1. Open the app, tap **Login**
2. A device code is shown (e.g. `A1B2C3`)
3. Go to `ai.connect-screen.com/device` in a browser and enter the code
4. Once approved, the app receives its API token and LLM config via the server
5. Config is persisted in AsyncStorage

After login, the app connects to the cloud via WebSocket to receive dispatched tasks.

## Trigger task via broadcast

The broadcast sends the task string directly to JS via `DeviceEventEmitter`:

```sh
adb shell "am broadcast -a ai.connct_screen.rn.EXECUTE_TASK -p ai.connct_screen.rn --es task 'open calculator'"
```

## Debugging

### View agent log file

Each task execution writes logs to `files/agent-log.txt` (cleared on each new task):

```sh
adb shell "run-as ai.connct_screen.rn cat files/agent-log.txt"
```

### View accessibility tree snapshots

Each `get_screen()` call saves the tree to `files/screens/screen_NNN.txt`:

```sh
adb shell "run-as ai.connct_screen.rn cat files/screens/screen_001.txt"
```

### View logcat

```sh
# Filter by tag
adb logcat -s A11yAgent
adb logcat -s ReactNativeJS

# Filter by app PID
adb logcat --pid=$(adb shell pidof ai.connct_screen.rn)
```

### Common errors

**Red screen: "TurboModule system assumes returnType == void iff the method is synchronous"**

Sync native methods (`isBlockingSynchronousMethod = true`) must have a non-void return type. The TurboModule interop layer uses return type to distinguish sync vs async:
- `void` return = async (must use `Promise` parameter)
- non-void return (`boolean`, `String`, `double`, etc.) = sync

If you need a sync method that logically returns nothing, return `boolean` (e.g. `return true`).

**Red screen: "Packager does not seem to be running"**

This happens if `debuggableVariants` in `build.gradle` includes `"debug"` (the default). With `debuggableVariants = []`, JS is bundled into the APK and Metro is not needed.

## Screenshot (`take_screenshot()`)

When the accessibility tree lacks text/desc (e.g. Feishu's `ImageView` buttons), the agent can capture a screenshot and send it to the LLM as a vision input.

How it works:
1. JS calls `take_screenshot()` which invokes `AccessibilityService.takeScreenshot()` (API 30+)
2. The async callback is blocked with `CountDownLatch` (10s timeout)
3. `HardwareBuffer` → `Bitmap` → scaled to 720px wide → JPEG q=80 → Base64
4. The base64 string is intercepted in `executeCode()` — JS gets a short placeholder, while the actual image is injected as a `user` message with `image_url` content type
5. `trimMessages()` keeps only the most recent screenshot to avoid context bloat

Requirements:
- `accessibility_service_config.xml` must have `android:canTakeScreenshot="true"`
- After changing this config, the accessibility service must be **re-enabled** in system settings
- API 30+ (Android 11+). Returns an error string on older devices
