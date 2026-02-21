import { useState, useEffect, useRef } from "react";
import useSWR from "swr";
import { Button } from "@cloudflare/kumo";
import {
  XIcon,
  FolderIcon,
  FileIcon,
  UploadIcon,
  DownloadIcon,
  TrashIcon,
  FolderPlusIcon,
  ArrowLeftIcon,
  ImageIcon,
  FileTextIcon,
  ChatCircleIcon,
  HardDriveIcon,
  CloudIcon,
  GitBranchIcon,
  LockIcon,
  PlusIcon,
  DatabaseIcon
} from "@phosphor-icons/react";
import {
  IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
  getExtension,
  joinPath
} from "../shared/file-utils";
import { Skeleton } from "./Skeleton";

interface FileEntry {
  name: string;
  isDirectory: boolean;
  size: number;
  mtime: string | null;
}

function formatSize(bytes: number): string {
  if (bytes === 0) return "\u2014";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const fileFetcher = async <T = any>(url: string): Promise<T> => {
  const res = await fetch(url);
  if (!res.ok) {
    const data = (await res.json()) as { error?: string };
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
};

// --- Fstab helpers ---

interface FstabEntry {
  device: string;
  mountPoint: string;
  type: string;
  options: string;
  dump: string;
  pass: string;
}

const PROTECTED_MOUNTS = new Set(["/etc", "/home/user", "/data"]);

function parseFstab(text: string): FstabEntry[] {
  const entries: FstabEntry[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const parts = trimmed.split(/\s+/);
    if (parts.length < 3) continue;
    entries.push({
      device: parts[0],
      mountPoint: parts[1],
      type: parts[2],
      options: parts[3] ?? "defaults",
      dump: parts[4] ?? "0",
      pass: parts[5] ?? "0"
    });
  }
  return entries;
}

function serializeFstab(entries: FstabEntry[], header?: string): string {
  const lines: string[] = [];
  if (header) lines.push(header);
  for (const e of entries) {
    lines.push(
      `${e.device}\t${e.mountPoint}\t${e.type}\t${e.options}\t${e.dump}\t${e.pass}`
    );
  }
  return lines.join("\n") + "\n";
}

const textFetcher = async (url: string): Promise<string> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
};

function mountTypeIcon(type: string) {
  switch (type) {
    case "git":
      return <GitBranchIcon size={16} className="text-orange-500" />;
    case "d1":
      return <DatabaseIcon size={16} className="text-blue-500" />;
    case "r2":
      return <CloudIcon size={16} className="text-purple-500" />;
    default:
      return <HardDriveIcon size={16} className="text-kumo-secondary" />;
  }
}

export function FileManagerPanel({
  open,
  onClose,
  onInsertFile
}: {
  open: boolean;
  onClose: () => void;
  onInsertFile?: (path: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<"files" | "mounts">("files");
  const [currentPath, setCurrentPath] = useState("/");
  const [preview, setPreview] = useState<{
    name: string;
    path: string;
    type: "text" | "image" | "other";
    content?: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showAddMount, setShowAddMount] = useState(false);
  const [newMount, setNewMount] = useState({
    type: "git",
    device: "",
    mountPoint: "/mnt/",
    options: ""
  });

  const { data, error, isValidating, mutate } = useSWR<{
    entries: FileEntry[];
  }>(
    open ? `/api/files/list?path=${encodeURIComponent(currentPath)}` : null,
    fileFetcher,
    { keepPreviousData: true }
  );
  const entries: FileEntry[] = data?.entries ?? [];

  const { data: fstabText, mutate: mutateFstab } = useSWR<string>(
    open && activeTab === "mounts"
      ? `/api/files/content?path=${encodeURIComponent("/etc/fstab")}`
      : null,
    textFetcher
  );
  const fstabEntries = fstabText ? parseFstab(fstabText) : [];

  const handleDeleteMount = async (entry: FstabEntry) => {
    if (!fstabText) return;
    const updated = fstabEntries.filter(
      (e) => !(e.device === entry.device && e.mountPoint === entry.mountPoint)
    );
    // Preserve comment lines from original text
    const commentLines = fstabText
      .split("\n")
      .filter((l) => l.trim().startsWith("#"))
      .join("\n");
    const newText = serializeFstab(updated, commentLines || undefined);
    await fetch(`/api/files/content?path=${encodeURIComponent("/etc/fstab")}`, {
      method: "PUT",
      body: newText
    });
    mutateFstab();
  };

  const handleAddMount = async () => {
    if (!newMount.device.trim() || !newMount.mountPoint.trim()) return;
    const entry: FstabEntry = {
      device: newMount.device.trim(),
      mountPoint: newMount.mountPoint.trim(),
      type: newMount.type,
      options: newMount.options.trim() || "defaults",
      dump: "0",
      pass: "0"
    };
    const currentEntries = fstabText ? parseFstab(fstabText) : [];
    const commentLines = (fstabText || "")
      .split("\n")
      .filter((l) => l.trim().startsWith("#"))
      .join("\n");
    const newText = serializeFstab(
      [...currentEntries, entry],
      commentLines || undefined
    );
    await fetch(`/api/files/content?path=${encodeURIComponent("/etc/fstab")}`, {
      method: "PUT",
      body: newText
    });
    setShowAddMount(false);
    setNewMount({ type: "git", device: "", mountPoint: "/mnt/", options: "" });
    mutateFstab();
  };

  // Clear preview when path changes
  useEffect(() => {
    setPreview(null);
  }, [currentPath]);

  const navigate = (name: string) => {
    setPreview(null);
    setCurrentPath((prev) => joinPath(prev, name));
  };

  const navigateToPath = (path: string) => {
    setPreview(null);
    setCurrentPath(path);
  };

  const handleFileClick = async (entry: FileEntry) => {
    if (entry.isDirectory) {
      navigate(entry.name);
      return;
    }
    const filePath = joinPath(currentPath, entry.name);
    const ext = getExtension(entry.name);

    if (IMAGE_EXTENSIONS.has(ext)) {
      setPreview({ name: entry.name, path: filePath, type: "image" });
    } else if (TEXT_EXTENSIONS.has(ext)) {
      try {
        const res = await fetch(
          `/api/files/content?path=${encodeURIComponent(filePath)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        setPreview({
          name: entry.name,
          path: filePath,
          type: "text",
          content: text
        });
      } catch {
        setPreview({ name: entry.name, path: filePath, type: "other" });
      }
    } else {
      setPreview({ name: entry.name, path: filePath, type: "other" });
    }
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      const filePath = joinPath(currentPath, file.name);
      const buffer = await file.arrayBuffer();
      await fetch(`/api/files/content?path=${encodeURIComponent(filePath)}`, {
        method: "PUT",
        body: buffer
      });
    }
    mutate();
  };

  const handleDelete = async (entry: FileEntry) => {
    const filePath = joinPath(currentPath, entry.name);
    const confirmed = window.confirm(
      `Delete ${entry.isDirectory ? "folder" : "file"} "${entry.name}"?`
    );
    if (!confirmed) return;
    await fetch(
      `/api/files?path=${encodeURIComponent(filePath)}&recursive=${entry.isDirectory ? "1" : "0"}`,
      { method: "DELETE" }
    );
    if (preview?.path === filePath) setPreview(null);
    mutate();
  };

  const handleNewFolder = async () => {
    const name = window.prompt("Folder name:");
    if (!name?.trim()) return;
    const folderPath = joinPath(currentPath, name.trim());
    await fetch("/api/files/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: folderPath })
    });
    mutate();
  };

  // Breadcrumb segments
  const pathSegments =
    currentPath === "/" ? [] : currentPath.split("/").filter(Boolean);

  if (!open) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 transition-opacity"
        onClick={onClose}
      />
      {/* Panel */}
      <div className="relative ml-auto flex h-full w-full max-w-md flex-col bg-kumo-base border-l border-kumo-line shadow-xl">
        {/* Header with tabs */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-kumo-line">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab("files")}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                activeTab === "files"
                  ? "bg-kumo-elevated text-kumo-default"
                  : "text-kumo-secondary hover:text-kumo-default"
              }`}
            >
              Files
            </button>
            <button
              onClick={() => setActiveTab("mounts")}
              className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                activeTab === "mounts"
                  ? "bg-kumo-elevated text-kumo-default"
                  : "text-kumo-secondary hover:text-kumo-default"
              }`}
            >
              Mounts
            </button>
            {isValidating && activeTab === "files" && (
              <div className="w-3 h-3 rounded-full border-2 border-kumo-secondary border-t-transparent animate-spin ml-1" />
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
          >
            <XIcon size={18} />
          </button>
        </div>

        {activeTab === "files" && (
          <>
            {/* Breadcrumb */}
            <div className="flex items-center gap-1 px-4 py-2 text-sm text-kumo-secondary overflow-x-auto border-b border-kumo-line">
              <button
                onClick={() => navigateToPath("/")}
                className="hover:text-kumo-default transition-colors shrink-0"
              >
                /
              </button>
              {pathSegments.map((seg, i) => {
                const segPath = "/" + pathSegments.slice(0, i + 1).join("/");
                const isLast = i === pathSegments.length - 1;
                return (
                  <span
                    key={segPath}
                    className="flex items-center gap-1 shrink-0"
                  >
                    <span className="text-kumo-line">/</span>
                    {isLast ? (
                      <span className="text-kumo-default font-medium">
                        {seg}
                      </span>
                    ) : (
                      <button
                        onClick={() => navigateToPath(segPath)}
                        className="hover:text-kumo-default transition-colors"
                      >
                        {seg}
                      </button>
                    )}
                  </span>
                );
              })}
            </div>

            {/* Toolbar */}
            <div className="flex items-center gap-2 px-4 py-2 border-b border-kumo-line">
              {currentPath !== "/" && currentPath !== "/home" && (
                <>
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<UploadIcon size={14} />}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Upload
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<FolderPlusIcon size={14} />}
                    onClick={handleNewFolder}
                  >
                    New Folder
                  </Button>
                </>
              )}
              {currentPath !== "/" && (
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<ArrowLeftIcon size={14} />}
                  onClick={() => {
                    const parent =
                      pathSegments.length <= 1
                        ? "/"
                        : "/" + pathSegments.slice(0, -1).join("/");
                    navigateToPath(parent);
                  }}
                >
                  Back
                </Button>
              )}
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                onChange={(e) => handleUpload(e.target.files)}
              />
            </div>

            {/* File listing */}
            <div className="flex-1 overflow-y-auto">
              {error && (
                <div className="p-4 text-sm text-red-500">{error.message}</div>
              )}
              {!error && !data && isValidating && (
                <div className="p-4 space-y-2">
                  {Array.from({ length: 5 }, (_, i) => (
                    <Skeleton key={i} className="h-10 w-full" />
                  ))}
                </div>
              )}
              {!error && entries.length === 0 && data && (
                <div className="p-4 text-sm text-kumo-secondary">
                  Empty directory
                </div>
              )}
              {entries.map((entry) => {
                const ext = getExtension(entry.name);
                const isImage = IMAGE_EXTENSIONS.has(ext);
                const isText = TEXT_EXTENSIONS.has(ext);
                const filePath = joinPath(currentPath, entry.name);

                return (
                  <div
                    key={entry.name}
                    className="group flex items-center gap-3 px-4 py-2 hover:bg-kumo-elevated cursor-pointer border-b border-kumo-line/50"
                    onClick={() => handleFileClick(entry)}
                  >
                    {/* Icon */}
                    <div className="shrink-0 text-kumo-secondary">
                      {entry.isDirectory ? (
                        <FolderIcon
                          size={18}
                          weight="fill"
                          className="text-amber-500"
                        />
                      ) : isImage ? (
                        <ImageIcon size={18} className="text-emerald-500" />
                      ) : isText ? (
                        <FileTextIcon size={18} className="text-blue-500" />
                      ) : (
                        <FileIcon size={18} />
                      )}
                    </div>
                    {/* Name + size */}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-kumo-default truncate">
                        {entry.name}
                      </div>
                      {!entry.isDirectory && (
                        <div className="text-xs text-kumo-secondary">
                          {formatSize(entry.size)}
                        </div>
                      )}
                    </div>
                    {/* Actions — visible on hover */}
                    <div className="shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {!entry.isDirectory && isImage && onInsertFile && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onInsertFile(filePath);
                          }}
                          className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-emerald-500 transition-colors"
                          title="Insert into chat"
                        >
                          <ChatCircleIcon size={14} />
                        </button>
                      )}
                      {!entry.isDirectory && (
                        <a
                          href={`/api/files/content?path=${encodeURIComponent(filePath)}`}
                          download={entry.name}
                          onClick={(e) => e.stopPropagation()}
                          className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-kumo-default transition-colors"
                          title="Download"
                        >
                          <DownloadIcon size={14} />
                        </a>
                      )}
                      {currentPath !== "/" && currentPath !== "/home" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(entry);
                          }}
                          className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-red-500 transition-colors"
                          title="Delete"
                        >
                          <TrashIcon size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Preview area */}
            {preview && (
              <div className="border-t border-kumo-line bg-kumo-elevated max-h-[40%] overflow-auto">
                <div className="flex items-center justify-between px-4 py-2 border-b border-kumo-line/50">
                  <span className="text-sm font-medium text-kumo-default truncate">
                    {preview.name}
                  </span>
                  <div className="flex items-center gap-1 shrink-0">
                    <a
                      href={`/api/files/content?path=${encodeURIComponent(preview.path)}`}
                      download={preview.name}
                      className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-kumo-default transition-colors"
                      title="Download"
                    >
                      <DownloadIcon size={14} />
                    </a>
                    <button
                      onClick={() => setPreview(null)}
                      className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-kumo-default transition-colors"
                    >
                      <XIcon size={14} />
                    </button>
                  </div>
                </div>
                <div className="p-4">
                  {preview.type === "image" && (
                    <img
                      src={`/api/files/content?path=${encodeURIComponent(preview.path)}`}
                      alt={preview.name}
                      className="max-w-full rounded"
                    />
                  )}
                  {preview.type === "text" && (
                    <pre className="text-xs text-kumo-default whitespace-pre-wrap break-all font-mono">
                      {preview.content}
                    </pre>
                  )}
                  {preview.type === "other" && (
                    <div className="text-sm text-kumo-secondary">
                      No preview available.{" "}
                      <a
                        href={`/api/files/content?path=${encodeURIComponent(preview.path)}`}
                        download={preview.name}
                        className="text-kumo-brand hover:underline"
                      >
                        Download
                      </a>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === "mounts" && (
          <>
            {/* Mounts toolbar */}
            <div className="flex items-center gap-2 px-4 py-2 border-b border-kumo-line">
              <Button
                variant="secondary"
                size="sm"
                icon={<PlusIcon size={14} />}
                onClick={() => setShowAddMount(true)}
              >
                Add Mount
              </Button>
            </div>

            {/* Add mount form */}
            {showAddMount && (
              <div className="px-4 py-3 border-b border-kumo-line bg-kumo-elevated space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-kumo-secondary w-16 shrink-0">
                    Type
                  </label>
                  <select
                    value={newMount.type}
                    onChange={(e) =>
                      setNewMount((m) => ({ ...m, type: e.target.value }))
                    }
                    className="flex-1 text-sm rounded-md border border-kumo-line bg-kumo-base text-kumo-default px-2 py-1"
                  >
                    <option value="git">git</option>
                    <option value="d1">d1</option>
                    <option value="r2">r2</option>
                    <option value="gdrive">gdrive</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-kumo-secondary w-16 shrink-0">
                    {newMount.type === "git" ? "URL" : "Device"}
                  </label>
                  <input
                    type="text"
                    value={newMount.device}
                    onChange={(e) =>
                      setNewMount((m) => ({ ...m, device: e.target.value }))
                    }
                    placeholder={
                      newMount.type === "git"
                        ? "https://github.com/user/repo"
                        : "device name"
                    }
                    className="flex-1 text-sm rounded-md border border-kumo-line bg-kumo-base text-kumo-default px-2 py-1"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-kumo-secondary w-16 shrink-0">
                    Mount
                  </label>
                  <input
                    type="text"
                    value={newMount.mountPoint}
                    onChange={(e) =>
                      setNewMount((m) => ({ ...m, mountPoint: e.target.value }))
                    }
                    className="flex-1 text-sm rounded-md border border-kumo-line bg-kumo-base text-kumo-default px-2 py-1"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-kumo-secondary w-16 shrink-0">
                    Options
                  </label>
                  <input
                    type="text"
                    value={newMount.options}
                    onChange={(e) =>
                      setNewMount((m) => ({ ...m, options: e.target.value }))
                    }
                    placeholder="defaults"
                    className="flex-1 text-sm rounded-md border border-kumo-line bg-kumo-base text-kumo-default px-2 py-1"
                  />
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <Button variant="primary" size="sm" onClick={handleAddMount}>
                    Save
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setShowAddMount(false);
                      setNewMount({
                        type: "git",
                        device: "",
                        mountPoint: "/mnt/",
                        options: ""
                      });
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}

            {/* Mount list */}
            <div className="flex-1 overflow-y-auto">
              {fstabEntries.length === 0 && fstabText !== undefined && (
                <div className="p-4 text-sm text-kumo-secondary">
                  No mounts configured
                </div>
              )}
              {fstabText === undefined && (
                <div className="p-4 space-y-2">
                  {Array.from({ length: 3 }, (_, i) => (
                    <Skeleton key={i} className="h-10 w-full" />
                  ))}
                </div>
              )}
              {fstabEntries.map((entry, i) => {
                const isProtected = PROTECTED_MOUNTS.has(entry.mountPoint);
                return (
                  <div
                    key={`${entry.mountPoint}-${i}`}
                    className="group flex items-center gap-3 px-4 py-2 hover:bg-kumo-elevated cursor-pointer border-b border-kumo-line/50"
                    onClick={() => {
                      setActiveTab("files");
                      navigateToPath(entry.mountPoint);
                    }}
                  >
                    <div className="shrink-0">{mountTypeIcon(entry.type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-kumo-default truncate flex items-center gap-1.5">
                        {entry.mountPoint}
                        {isProtected && (
                          <LockIcon
                            size={12}
                            className="text-kumo-secondary shrink-0"
                          />
                        )}
                      </div>
                      <div className="text-xs text-kumo-secondary truncate">
                        <span className="inline-block bg-kumo-elevated text-kumo-secondary rounded px-1 py-0.5 text-[10px] font-medium mr-1">
                          {entry.type}
                        </span>
                        {entry.device}
                      </div>
                    </div>
                    {!isProtected && (
                      <div className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteMount(entry);
                          }}
                          className="p-1 rounded hover:bg-kumo-base text-kumo-secondary hover:text-red-500 transition-colors"
                          title="Remove mount"
                        >
                          <TrashIcon size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
