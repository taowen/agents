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
| `SelectToSpeakService.java` | Core accessibility service (reads UI tree, performs clicks/scrolls) |
| `TaskReceiver.java` | BroadcastReceiver that emits `onTaskReceived` event to RN |

## Build & Deploy

```sh
# Bundle JS (verify TS compiles)
cd examples/rn
npx react-native bundle --platform android --dev false --entry-file index.js \
  --bundle-output /tmp/rn-test-bundle.js --assets-dest /tmp/rn-test-assets

# Build APK
cd android && ./gradlew assembleDebug

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

## Trigger task via broadcast

```sh
adb shell am broadcast -a ai.connct_screen.rn.EXECUTE_TASK \
  --es task "open calculator" \
  --es api_url "https://api.openai.com/v1" \
  --es api_key "sk-..." \
  --es model "gpt-4o"
```

If the app already has LLM config saved (via the in-app config panel), the broadcast only needs `--es task "..."`.

## Debugging

### View app logs

```sh
# Filter by app PID
PID=$(adb shell pidof ai.connct_screen.rn)
adb logcat --pid=$PID

# Filter by tag
adb logcat -s A11yAgent
adb logcat -s ReactNativeJS

# Search for errors in all recent logs from the app process
adb logcat -d | grep "$PID" | grep -iE "error|exception|fatal"
```

### Common errors

**Red screen: "TurboModule system assumes returnType == void iff the method is synchronous"**

Sync native methods (`isBlockingSynchronousMethod = true`) must have a non-void return type. The TurboModule interop layer uses return type to distinguish sync vs async:
- `void` return = async (must use `Promise` parameter)
- non-void return (`boolean`, `String`, `double`, etc.) = sync

If you need a sync method that logically returns nothing, return `boolean` (e.g. `return true`).

**Red screen: "Packager does not seem to be running"**

Debug builds try to connect to Metro dev server. Either:
- Run `npm start` in the project root, or
- Build a release APK: `./gradlew assembleRelease`

## LLM Config

Config is stored in `android/llm-config.json` (gitignored) and copied into APK assets at build time. Can also be configured in-app via the "LLM Config" panel, which persists to AsyncStorage.

```json
{
  "baseURL": "https://api.openai.com/v1",
  "apiKey": "sk-...",
  "model": "gpt-4o"
}
```
