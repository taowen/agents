package ai.connct_screen.rn;

import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GPT-2 style byte-level BPE tokenizer for Qwen2.
 *
 * Loads vocab.json (token-to-id map) and merges.txt (merge rules) from the model directory,
 * then tokenizes text into BPE token IDs.
 *
 * Also applies the Qwen2 chat template for TTS:
 *   [im_start, assistant, \n, ...text_tokens..., im_end, \n, im_start, assistant, \n]
 */
public class BpeTokenizer {

    private static final String TAG = "BpeTokenizer";

    // Special token IDs
    private static final int TOKEN_IM_START = 151644;
    private static final int TOKEN_IM_END   = 151645;
    private static final int TOKEN_ASSISTANT = 77091;
    private static final int TOKEN_NEWLINE   = 198;   // "Ċ" in GPT-2 byte encoding

    // GPT-2 byte-to-unicode mapping
    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final Map<Character, Integer> UNICODE_TO_BYTE = new HashMap<>();

    static {
        // Build the GPT-2 byte-to-unicode mapping.
        // Printable bytes map to themselves; others get shifted to avoid control chars.
        int n = 0;
        // Range: '!' (33) to '~' (126)
        for (int b = 33; b <= 126; b++) {
            BYTE_TO_UNICODE[b] = (char) b;
            n++;
        }
        // Range: '¡' (161) to '¬' (172)
        for (int b = 161; b <= 172; b++) {
            BYTE_TO_UNICODE[b] = (char) b;
            n++;
        }
        // Range: '®' (174) to 'ÿ' (255)
        for (int b = 174; b <= 255; b++) {
            BYTE_TO_UNICODE[b] = (char) b;
            n++;
        }
        // Remaining bytes (0-32, 127-160, 173) get mapped to 256+
        int offset = 256;
        for (int b = 0; b < 256; b++) {
            if (BYTE_TO_UNICODE[b] == 0 && b != 0) {
                // hasn't been set yet
                BYTE_TO_UNICODE[b] = (char) offset;
                offset++;
            } else if (b == 0) {
                // byte 0 also hasn't been set
                BYTE_TO_UNICODE[b] = (char) offset;
                offset++;
            }
        }
        // Fix: the code above doesn't handle byte 0 correctly since char 0 == default.
        // Let's rebuild properly.
        // Clear and redo
        boolean[] set = new boolean[256];
        for (int b = 33; b <= 126; b++) { BYTE_TO_UNICODE[b] = (char) b; set[b] = true; }
        for (int b = 161; b <= 172; b++) { BYTE_TO_UNICODE[b] = (char) b; set[b] = true; }
        for (int b = 174; b <= 255; b++) { BYTE_TO_UNICODE[b] = (char) b; set[b] = true; }
        offset = 256;
        for (int b = 0; b < 256; b++) {
            if (!set[b]) {
                BYTE_TO_UNICODE[b] = (char) offset;
                offset++;
            }
        }

        // Build reverse mapping
        for (int b = 0; b < 256; b++) {
            UNICODE_TO_BYTE.put(BYTE_TO_UNICODE[b], b);
        }
    }

    // Token -> ID map
    private final Map<String, Integer> vocab;
    // Merge pair -> priority (lower = higher priority)
    private final Map<String, Integer> mergeRanks;

    // GPT-2 pre-tokenization regex (matches words, contractions, numbers, etc.)
    private final Pattern pretokenizePattern;

    private BpeTokenizer(Map<String, Integer> vocab, Map<String, Integer> mergeRanks) {
        this.vocab = vocab;
        this.mergeRanks = mergeRanks;
        // Qwen2 / GPT-2 style pre-tokenization pattern
        this.pretokenizePattern = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d" +
            "|[\\p{L}]+" +
            "|[\\p{N}]+" +
            "| ?[^\\s\\p{L}\\p{N}]+" +
            "|\\s+"
        );
    }

    /**
     * Load tokenizer from model directory containing vocab.json and merges.txt.
     */
    public static BpeTokenizer load(String modelDir) {
        try {
            // Load vocab.json
            File vocabFile = new File(modelDir, "vocab.json");
            String vocabJson = readFileToString(vocabFile);
            JSONObject vocabObj = new JSONObject(vocabJson);
            Map<String, Integer> vocab = new HashMap<>();
            Iterator<String> keys = vocabObj.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                vocab.put(key, vocabObj.getInt(key));
            }
            Log.i(TAG, "Loaded vocab: " + vocab.size() + " tokens");

            // Load merges.txt
            File mergesFile = new File(modelDir, "merges.txt");
            Map<String, Integer> mergeRanks = new HashMap<>();
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(mergesFile), "UTF-8"));
            String line;
            int rank = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#version")) continue;
                mergeRanks.put(line, rank);
                rank++;
            }
            reader.close();
            Log.i(TAG, "Loaded merges: " + mergeRanks.size() + " rules");

            return new BpeTokenizer(vocab, mergeRanks);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load tokenizer", e);
            return null;
        }
    }

    /**
     * Tokenize text and wrap in Qwen2 chat template for TTS.
     * Returns comma-separated token IDs string ready for the C engine.
     *
     * Template: [im_start, assistant, \n, TEXT..., im_end, \n, im_start, assistant, \n]
     */
    public String tokenizeForTts(String text) {
        List<Integer> textTokens = encode(text);

        List<Integer> result = new ArrayList<>();
        // Prefix: <|im_start|> assistant \n
        result.add(TOKEN_IM_START);
        result.add(TOKEN_ASSISTANT);
        result.add(TOKEN_NEWLINE);
        // Text tokens
        result.addAll(textTokens);
        // Suffix: <|im_end|> \n <|im_start|> assistant \n
        result.add(TOKEN_IM_END);
        result.add(TOKEN_NEWLINE);
        result.add(TOKEN_IM_START);
        result.add(TOKEN_ASSISTANT);
        result.add(TOKEN_NEWLINE);

        // Convert to comma-separated string
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < result.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(result.get(i));
        }
        return sb.toString();
    }

    /**
     * Encode text into BPE token IDs.
     */
    public List<Integer> encode(String text) {
        List<Integer> tokens = new ArrayList<>();

        Matcher matcher = pretokenizePattern.matcher(text);
        while (matcher.find()) {
            String word = matcher.group();
            // Convert word bytes to GPT-2 unicode characters
            String bpeWord = bytesToBpeString(word);
            // Apply BPE merges
            List<String> bpeTokens = bpe(bpeWord);
            for (String token : bpeTokens) {
                Integer id = vocab.get(token);
                if (id != null) {
                    tokens.add(id);
                } else {
                    // Fallback: encode each character individually
                    for (int i = 0; i < token.length(); i++) {
                        String ch = String.valueOf(token.charAt(i));
                        Integer chId = vocab.get(ch);
                        if (chId != null) {
                            tokens.add(chId);
                        }
                    }
                }
            }
        }

        return tokens;
    }

    /**
     * Convert a string to its GPT-2 byte-encoded representation.
     */
    private String bytesToBpeString(String text) {
        byte[] bytes;
        try {
            bytes = text.getBytes("UTF-8");
        } catch (Exception e) {
            bytes = text.getBytes();
        }
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(BYTE_TO_UNICODE[b & 0xFF]);
        }
        return sb.toString();
    }

    /**
     * Apply BPE merges to a string of GPT-2 unicode characters.
     * Returns the list of BPE tokens.
     */
    private List<String> bpe(String token) {
        if (token.length() <= 1) {
            List<String> result = new ArrayList<>();
            result.add(token);
            return result;
        }

        // Start with individual characters
        List<String> word = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            word.add(String.valueOf(token.charAt(i)));
        }

        while (word.size() > 1) {
            // Find the pair with the lowest merge rank
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;

            for (int i = 0; i < word.size() - 1; i++) {
                String pair = word.get(i) + " " + word.get(i + 1);
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0) break; // No more merges

            // Merge the best pair
            String merged = word.get(bestIdx) + word.get(bestIdx + 1);
            List<String> newWord = new ArrayList<>();
            for (int i = 0; i < word.size(); i++) {
                if (i == bestIdx) {
                    newWord.add(merged);
                    i++; // skip next element (it was merged)
                } else {
                    newWord.add(word.get(i));
                }
            }
            word = newWord;
        }

        return word;
    }

    private static String readFileToString(File file) throws Exception {
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int) file.length()];
        fis.read(data);
        fis.close();
        return new String(data, "UTF-8");
    }
}
