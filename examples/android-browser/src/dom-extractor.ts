/**
 * dom-extractor.ts
 *
 * Self-contained IIFE injected into the visible WebView to extract an
 * interaction tree from the DOM.  Assigns numeric IDs to interactive
 * elements and stores references in window.__agentElements for later
 * click / type actions.
 *
 * Returns a formatted string describing the page structure.
 */

(function extractDom(): string {
  const INTERACTIVE_TAGS = new Set([
    "A",
    "BUTTON",
    "INPUT",
    "SELECT",
    "TEXTAREA",
    "DETAILS",
    "SUMMARY"
  ]);
  const INTERACTIVE_ROLES = new Set([
    "button",
    "link",
    "menuitem",
    "tab",
    "checkbox",
    "radio",
    "switch",
    "option",
    "combobox",
    "textbox",
    "searchbox",
    "slider",
    "spinbutton"
  ]);
  const SKIP_TAGS = new Set([
    "SCRIPT",
    "STYLE",
    "NOSCRIPT",
    "SVG",
    "PATH",
    "META",
    "LINK",
    "BR",
    "HR",
    "WBR",
    "TEMPLATE"
  ]);
  const TEXT_TAGS = new Set([
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "P",
    "LABEL",
    "LI",
    "TD",
    "TH",
    "CAPTION",
    "FIGCAPTION",
    "LEGEND",
    "BLOCKQUOTE",
    "PRE"
  ]);
  const MAX_ELEMENTS = 500;

  const elements: Record<number, Element> = {};
  let nextId = 1;
  const lines: string[] = [];

  // Page info
  lines.push("Page: " + document.title);
  lines.push("URL: " + location.href);
  lines.push("");

  function isVisible(el: Element): boolean {
    if (!(el instanceof HTMLElement)) return true;
    const style = getComputedStyle(el);
    if (style.display === "none") return false;
    if (style.visibility === "hidden") return false;
    if (style.opacity === "0") return false;
    // Check if element has zero size
    if (el.offsetWidth === 0 && el.offsetHeight === 0) return false;
    return true;
  }

  function isInteractive(el: Element): boolean {
    if (INTERACTIVE_TAGS.has(el.tagName)) return true;
    const role = el.getAttribute("role");
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.hasAttribute("onclick") || el.hasAttribute("onmousedown"))
      return true;
    if ((el as HTMLElement).contentEditable === "true") return true;
    // Check for tabindex that makes element focusable
    if (el.hasAttribute("tabindex") && el.getAttribute("tabindex") !== "-1")
      return true;
    return false;
  }

  function getDirectText(el: Element): string {
    let text = "";
    for (const child of el.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        text += child.textContent || "";
      }
    }
    return text.trim();
  }

  function describeElement(el: Element): string | null {
    const tag = el.tagName.toLowerCase();

    if (isInteractive(el)) {
      const id = nextId++;
      elements[id] = el;

      const parts: string[] = [];
      const role = el.getAttribute("role");

      // Type description
      if (tag === "a") {
        parts.push("link");
      } else if (tag === "button" || role === "button") {
        parts.push("button");
      } else if (tag === "input") {
        const inputType = (el as HTMLInputElement).type || "text";
        parts.push("input[" + inputType + "]");
      } else if (tag === "select") {
        parts.push("select");
      } else if (tag === "textarea") {
        parts.push("textarea");
      } else if (role) {
        parts.push(role);
      } else {
        parts.push(tag);
      }

      // Text content (fall back to innerText for nested content like <button><span>Search</span></button>)
      const text =
        getDirectText(el) ||
        (el as HTMLElement).innerText?.trim()?.substring(0, 80) ||
        el.getAttribute("aria-label") ||
        el.getAttribute("title") ||
        (el as HTMLInputElement).value ||
        "";
      if (text) {
        parts.push('"' + text.substring(0, 80) + '"');
      }

      // Extra attributes
      if (tag === "a") {
        const href = el.getAttribute("href");
        if (href && href !== "#") {
          parts.push('href="' + href.substring(0, 100) + '"');
        }
      }
      if (tag === "input" || tag === "textarea") {
        const placeholder = el.getAttribute("placeholder");
        if (placeholder) {
          parts.push('placeholder="' + placeholder.substring(0, 60) + '"');
        }
        const value = (el as HTMLInputElement).value;
        if (value) {
          parts.push('value="' + value.substring(0, 60) + '"');
        }
      }
      if (tag === "select") {
        const selected = (el as HTMLSelectElement).selectedOptions;
        if (selected.length > 0) {
          parts.push('selected="' + selected[0].text.substring(0, 60) + '"');
        }
      }
      if (el.getAttribute("aria-expanded")) {
        parts.push("expanded=" + el.getAttribute("aria-expanded"));
      }
      if ((el as HTMLInputElement).checked) {
        parts.push("checked");
      }
      if ((el as HTMLInputElement).disabled) {
        parts.push("disabled");
      }

      return "[" + id + "] " + parts.join(" ");
    }

    // Text-bearing elements (for context, no ID)
    if (TEXT_TAGS.has(el.tagName)) {
      const text =
        getDirectText(el) || (el as HTMLElement).innerText?.trim() || "";
      if (text && text.length > 2) {
        const tagLabel = tag.match(/^h[1-6]$/) ? "heading" : "text";
        return tagLabel + ' "' + text.substring(0, 150) + '"';
      }
    }

    return null;
  }

  function walk(el: Element): void {
    if (nextId > MAX_ELEMENTS) return;
    if (SKIP_TAGS.has(el.tagName)) return;
    if (!isVisible(el)) return;

    const desc = describeElement(el);
    if (desc) {
      lines.push(desc);
    }

    // Walk children
    for (const child of el.children) {
      walk(child);
    }
  }

  walk(document.body || document.documentElement);

  // Store element references for later use by click/type
  (window as any).__agentElements = elements;

  // Store result on global so Java can read it via evaluateJavascript
  (window as any).__agentDomTree = lines.join("\n");
  return (window as any).__agentDomTree;
})();
