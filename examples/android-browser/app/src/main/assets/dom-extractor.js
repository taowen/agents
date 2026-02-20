"use strict";
(() => {
  // src/dom-extractor.ts
  (function extractDom() {
    const INTERACTIVE_TAGS = /* @__PURE__ */ new Set([
      "A",
      "BUTTON",
      "INPUT",
      "SELECT",
      "TEXTAREA",
      "DETAILS",
      "SUMMARY"
    ]);
    const INTERACTIVE_ROLES = /* @__PURE__ */ new Set([
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
    const SKIP_TAGS = /* @__PURE__ */ new Set([
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
    const TEXT_TAGS = /* @__PURE__ */ new Set([
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
    const elements = {};
    let nextId = 1;
    const lines = [];
    lines.push("Page: " + document.title);
    lines.push("URL: " + location.href);
    lines.push("");
    function isVisible(el) {
      if (!(el instanceof HTMLElement)) return true;
      const style = getComputedStyle(el);
      if (style.display === "none") return false;
      if (style.visibility === "hidden") return false;
      if (style.opacity === "0") return false;
      if (el.offsetWidth === 0 && el.offsetHeight === 0) return false;
      return true;
    }
    function isInteractive(el) {
      if (INTERACTIVE_TAGS.has(el.tagName)) return true;
      const role = el.getAttribute("role");
      if (role && INTERACTIVE_ROLES.has(role)) return true;
      if (el.hasAttribute("onclick") || el.hasAttribute("onmousedown"))
        return true;
      if (el.contentEditable === "true") return true;
      if (el.hasAttribute("tabindex") && el.getAttribute("tabindex") !== "-1")
        return true;
      return false;
    }
    function getDirectText(el) {
      let text = "";
      for (const child of el.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
          text += child.textContent || "";
        }
      }
      return text.trim();
    }
    function describeElement(el) {
      const tag = el.tagName.toLowerCase();
      if (isInteractive(el)) {
        const id = nextId++;
        elements[id] = el;
        const parts = [];
        const role = el.getAttribute("role");
        if (tag === "a") {
          parts.push("link");
        } else if (tag === "button" || role === "button") {
          parts.push("button");
        } else if (tag === "input") {
          const inputType = el.type || "text";
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
        const text =
          getDirectText(el) ||
          el.innerText?.trim()?.substring(0, 80) ||
          el.getAttribute("aria-label") ||
          el.getAttribute("title") ||
          el.value ||
          "";
        if (text) {
          parts.push('"' + text.substring(0, 80) + '"');
        }
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
          const value = el.value;
          if (value) {
            parts.push('value="' + value.substring(0, 60) + '"');
          }
        }
        if (tag === "select") {
          const selected = el.selectedOptions;
          if (selected.length > 0) {
            parts.push('selected="' + selected[0].text.substring(0, 60) + '"');
          }
        }
        if (el.getAttribute("aria-expanded")) {
          parts.push("expanded=" + el.getAttribute("aria-expanded"));
        }
        if (el.checked) {
          parts.push("checked");
        }
        if (el.disabled) {
          parts.push("disabled");
        }
        return "[" + id + "] " + parts.join(" ");
      }
      if (TEXT_TAGS.has(el.tagName)) {
        const text = getDirectText(el) || el.innerText?.trim() || "";
        if (text && text.length > 2) {
          const tagLabel = tag.match(/^h[1-6]$/) ? "heading" : "text";
          return tagLabel + ' "' + text.substring(0, 150) + '"';
        }
      }
      return null;
    }
    function walk(el) {
      if (nextId > MAX_ELEMENTS) return;
      if (SKIP_TAGS.has(el.tagName)) return;
      if (!isVisible(el)) return;
      const desc = describeElement(el);
      if (desc) {
        lines.push(desc);
      }
      for (const child of el.children) {
        walk(child);
      }
    }
    walk(document.body || document.documentElement);
    window.__agentElements = elements;
    window.__agentDomTree = lines.join("\n");
    return window.__agentDomTree;
  })();
})();
