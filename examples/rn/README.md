# RN Agent

Android Accessibility Service agent powered by React Native + Hermes.

LLM generates JavaScript code, which is `eval()`'d directly in Hermes. The JS code calls synchronous native methods (`@ReactMethod(isBlockingSynchronousMethod = true)`) to interact with the accessibility tree.

## Architecture

```
AgentLoop.ts  ──>  executeCode(code)  [eval() in Hermes]
                     │
                     ├── globalThis.get_screen()   ──>  AccessibilityBridge.getScreen()  [sync JSI]
                     ├── globalThis.click(target)   ──>  clickByText / clickByDesc / clickByCoords
                     ├── globalThis.long_click(...)  ──>  longClickByText / longClickByDesc / longClickByCoords
                     ├── globalThis.scroll(dir)     ──>  scrollScreen()
                     ├── globalThis.type_text(...)   ──>  typeText()
                     ├── globalThis.take_screenshot()──>  takeScreenshotSync() → base64 JPEG
                     ├── globalThis.sleep(ms)       ──>  sleepMs()
                     ├── ...                        ──>  pressHome / pressBack / launchApp / listApps / ...
                     │
                     └── SelectToSpeakService  (Android Accessibility Service)
```

Key files:

| File | Role |
|------|------|
| `src/agentGlobals.ts` | Registers global functions on `globalThis`, exports `executeCode()` |
| `src/AgentLoop.ts` | LLM conversation loop, calls `executeCode()` for tool execution |
| `src/LlmClient.ts` | HTTP client for OpenAI-compatible chat completions API |
| `src/App.tsx` | UI: task input, log viewer, LLM config, broadcast event listener |
| `AccessibilityBridgeModule.java` | Sync native methods bridging to SelectToSpeakService |
| `SelectToSpeakService.java` | Core accessibility service (reads UI tree, performs clicks/scrolls, takes screenshots) |
| `TaskReceiver.java` | BroadcastReceiver that emits `onTaskReceived` event to RN |

## Build & Deploy

JS is bundled into the debug APK (no Metro dev server needed — `debuggableVariants = []` in build.gradle).

```sh
cd examples/rn/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

## Trigger task via broadcast

The broadcast sends the task string directly to JS via `DeviceEventEmitter` (no polling, no JSON wrapping). **Important:** use `-p` to target the package explicitly and quote the whole `am` command for `adb shell` so multi-word tasks aren't split:

```sh
adb shell "am broadcast -a ai.connct_screen.rn.EXECUTE_TASK -p ai.connct_screen.rn --es task 'open calculator'"
```

The app uses LLM config from the in-app config panel (AsyncStorage) or the bundled `android/llm-config.json` asset.

## Debugging

### View agent log file

Each task execution writes logs to `files/agent-log.txt` (cleared on each new task). Read it via:

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

## LLM Config

Config is stored in `android/llm-config.json` (gitignored) and copied into APK assets at build time. Can also be configured in-app via the "LLM Config" panel, which persists to AsyncStorage.

```json
{
  "baseURL": "https://api.openai.com/v1",
  "apiKey": "sk-...",
  "model": "gpt-4o"
}
```
