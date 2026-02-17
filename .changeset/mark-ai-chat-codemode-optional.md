---
"agents": patch
"@cloudflare/ai-chat": patch
"@cloudflare/codemode": patch
"hono-agents": patch
---

Widen peer dependency ranges across packages to prevent cascading major bumps during 0.x minor releases. Mark `@cloudflare/ai-chat` and `@cloudflare/codemode` as optional peer dependencies of `agents` to fix unmet peer dependency warnings during installation.
