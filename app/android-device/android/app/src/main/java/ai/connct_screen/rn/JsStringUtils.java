package ai.connct_screen.rn;

/**
 * Shared JS/JSON string escaping utilities.
 */
public final class JsStringUtils {

    private JsStringUtils() {}

    /**
     * Escape a string for embedding inside JS double quotes.
     * Does NOT add surrounding quotes.
     */
    public static String escapeForJS(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * Escape a string and wrap it in double quotes for JS embedding.
     */
    public static String quoteForJS(String s) {
        return "\"" + escapeForJS(s) + "\"";
    }
}
