# App

This directory contains the full application — a Cloudflare Worker server paired with an Android device agent.

## Projects

| Directory | Description |
|-----------|-------------|
| `ai-chat/` | Cloudflare Worker server with sandboxed bash, D1/R2 storage, device hub, and browser tool |
| `android-device/` | Android device agent that connects to ai-chat via device auth + WebSocket |

## How they work together

1. The Android agent authenticates with the server using a device auth flow (similar to OAuth2 Device Authorization)
2. Once linked, the server can dispatch tasks to the agent via WebSocket
3. The agent executes tasks by generating JavaScript code that controls the Android device through accessibility APIs
4. Results and screenshots are sent back to the server for the LLM to process
