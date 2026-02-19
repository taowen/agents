# vfs

Virtual filesystem layer for just-bash. All filesystem adapters, mount/boot logic, and git commands live here.

## Filesystem Adapters

- **D1FsAdapter** — Cloudflare D1 (SQL) backed filesystem
- **R2FsAdapter** — Cloudflare R2 (object storage) backed filesystem
- **GoogleDriveFsAdapter** — Google Drive backed filesystem
- **GitFs** — Git repository filesystem (read-write, auto commit & push via R2 overlay)

## Bootstrap

`bootFilesystem(mountableFs, options)` — two-phase boot:

1. Mount `/etc` (so fstab can be read)
2. Read `/etc/fstab`, migrate legacy entries if needed, mount all entries

```ts
import { bootFilesystem } from "vfs";

await bootFilesystem(mountableFs, {
  etcFs: d1Adapter,       // filesystem backing /etc (typically D1FsAdapter)
  etcFsType: "d1",        // fstab type label for /etc (default: "d1")
  fsTypeRegistry: { ... },
  r2Bucket,
  userId,
});
```

## Commands

### mount / umount

`createMountCommands(mountableFs, options?)` provides `mount` and `umount` commands for just-bash:

- `mount` — list current mounts
- `mount -t git [-o ref=main,depth=1] <url> <mountpoint>` — mount a git repo (auto-persisted to `/etc/fstab`)
- `mount -t d1 <device> <mountpoint>` — mount D1 filesystem
- `mount -t r2 <device> <mountpoint>` — mount R2 filesystem
- `mount -t gdrive <device> <mountpoint>` — mount Google Drive
- `umount <mountpoint>` — unmount and auto-remove from `/etc/fstab`

Mount types beyond `git` are resolved via the `fsTypeRegistry` in `MountOptions`.

### git

`createGitCommands(mountableFs, options?)` provides the `git` command for just-bash:

- `git clone <url> [<dir>]` — clone a repo (creates a new git mount)
- `git status` — show working tree status
- `git diff` — show changes
- `git log` — show commit history
- `git commit -m "msg"` — commit staged changes
- `git push` — push to remote
- `git pull` — pull from remote
- `git branch` — show current branch
- `git remote` — show remotes
- `git show` — show commit details
- `git rev-parse` — parse git references

Git commands auto-detect the repo by finding the GitFs mount whose mount point is a prefix of the current working directory.

## Fstab

`parseFstab(content)` parses `/etc/fstab` format (standard 6-column: device, mountpoint, type, options, dump, pass). `mount` and `umount` commands automatically manage fstab entries — no manual editing needed.

Default fstab:
```
none  /etc        d1  defaults  0  0
none  /home/user  d1  defaults  0  0
none  /data       r2  defaults  0  0
```

## Git Credentials

`parseGitCredentials` / `findCredential` read `/etc/git-credentials` for authentication.

## GitHub OAuth

`createGitHubOAuthRoutes` provides Worker-side routes (`/oauth/github/*`) for GitHub OAuth flow, storing tokens into `/etc/git-credentials` on the DO.
