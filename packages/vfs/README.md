# vfs

Virtual filesystem layer for just-bash. All filesystem adapters and filesystem operation commands live here.

## Filesystem Adapters

- **D1FsAdapter** — Cloudflare D1 (SQL) backed filesystem
- **R2FsAdapter** — Cloudflare R2 (object storage) backed filesystem
- **AgentFsAdapter** — AgentFS backed filesystem
- **GitFs** — Git repository filesystem (read-write, auto commit & push)

## Commands

`createMountCommands(mountableFs, agentFs?)` provides `mount` and `umount` commands for just-bash:

- `mount` — list current mounts
- `mount -t git [-o ref=main,depth=1] <url> <mountpoint>` — mount a git repo (auto-persisted to `/etc/fstab`)
- `mount -t agentfs <device> <mountpoint>` — mount an AgentFS path (requires `agentFs` parameter)
- `umount <mountpoint>` — unmount and auto-remove from `/etc/fstab`

Applications (like ai-chat) should use `createMountCommands` from this package rather than implementing their own mount/umount commands.

## Fstab

`parseFstab(content)` parses `/etc/fstab` format. `mount` and `umount` commands automatically manage fstab entries — no manual editing needed.

## Git Credentials

`parseGitCredentials` / `findCredential` read `/etc/git-credentials` for authentication. GitHub OAuth integration is provided via `createGitHubOAuthRoutes`.
