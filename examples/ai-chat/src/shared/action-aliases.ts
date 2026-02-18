/**
 * Single source of truth for action-name aliases.
 * Handles common LLM misspellings / shorthand variants.
 */

const actionAliases: Record<string, string> = {
  press_key: "key_press",
  keypress: "key_press",
  move: "mouse_move",
  move_mouse: "mouse_move",
  focus: "focus_window",
  activate: "focus_window",
  activate_window: "focus_window",
  resize: "resize_window",
  move_window: "resize_window",
  minimize: "minimize_window",
  maximize: "maximize_window",
  restore: "restore_window",
  list: "list_windows",
  double_click: "click"
};

/**
 * Normalize an action name using the alias map.
 * Returns the canonical action name.
 */
export function normalizeAction(action: string): string {
  return actionAliases[action] ?? action;
}

/**
 * Returns true if the original (pre-normalized) action was "double_click".
 */
export function isDoubleClickAlias(action: string): boolean {
  return action === "double_click";
}
