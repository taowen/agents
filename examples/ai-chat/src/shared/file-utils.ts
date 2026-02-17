export const IMAGE_EXTENSIONS = new Set([
  "png",
  "jpg",
  "jpeg",
  "gif",
  "webp",
  "svg"
]);

export const TEXT_EXTENSIONS = new Set([
  "txt",
  "md",
  "json",
  "js",
  "ts",
  "css",
  "html",
  "csv",
  "sh",
  "xml",
  "yaml",
  "yml",
  "toml",
  "env",
  "log",
  "tsx",
  "jsx"
]);

export const MIME_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  svg: "image/svg+xml",
  pdf: "application/pdf",
  json: "application/json",
  js: "text/javascript",
  ts: "text/plain",
  css: "text/css",
  html: "text/html",
  md: "text/markdown",
  txt: "text/plain",
  csv: "text/csv",
  sh: "text/plain",
  xml: "application/xml"
};

export function getExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
}

export function joinPath(dir: string, name: string): string {
  return dir === "/" ? `/${name}` : `${dir}/${name}`;
}
