package ai.connct_screen.com;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.util.Log;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import org.mozilla.javascript.BaseFunction;
import org.mozilla.javascript.ContextFactory;
import org.mozilla.javascript.NativeObject;
import org.mozilla.javascript.Scriptable;
import org.mozilla.javascript.ScriptableObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class JsEngine {

    private static final String TAG = "A11yAgent";
    private static final long TIMEOUT_MS = 30_000;
    private static final int MAX_GET_SCREEN_PER_EXEC = 5;

    private final ScriptableObject scope;
    private final SelectToSpeakService service;
    private final File screensDir;
    private final TimeLimitContextFactory contextFactory;
    private StringBuilder actionLog;
    private int screenCounter = 0;
    private int getScreenCount = 0;

    /**
     * ContextFactory that installs an instruction observer to detect pure-JS
     * infinite loops (e.g. while(true){}) that never call our global functions.
     */
    private static class TimeLimitContextFactory extends ContextFactory {
        volatile long deadline;

        @Override
        protected org.mozilla.javascript.Context makeContext() {
            org.mozilla.javascript.Context cx = super.makeContext();
            cx.setOptimizationLevel(-1);
            cx.setInstructionObserverThreshold(10_000);
            return cx;
        }

        @Override
        protected void observeInstructionCount(org.mozilla.javascript.Context cx, int instructionCount) {
            if (System.currentTimeMillis() > deadline) {
                throw new Error("Script execution timeout (instruction observer)");
            }
        }
    }

    public JsEngine(SelectToSpeakService service, Context appContext, File screensDir) {
        this.service = service;
        this.screensDir = screensDir;
        this.contextFactory = new TimeLimitContextFactory();
        org.mozilla.javascript.Context cx = contextFactory.enterContext();
        try {
            this.scope = cx.initStandardObjects();
            registerFunctions(cx);
        } finally {
            org.mozilla.javascript.Context.exit();
        }
    }

    public void resetScreenCounter() {
        screenCounter = 0;
    }

    public String execute(String code) {
        actionLog = new StringBuilder();
        getScreenCount = 0;
        Thread execThread = Thread.currentThread();

        // Set deadline for instruction observer
        contextFactory.deadline = System.currentTimeMillis() + TIMEOUT_MS;

        // Watchdog: interrupt execution thread after TIMEOUT_MS (fallback for blocking I/O)
        ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor();
        ScheduledFuture<?> timeoutFuture = watchdog.schedule(() -> {
            Log.w(TAG, "[TIMEOUT] JS execution exceeded " + TIMEOUT_MS + "ms, interrupting");
            execThread.interrupt();
        }, TIMEOUT_MS, TimeUnit.MILLISECONDS);

        org.mozilla.javascript.Context cx = contextFactory.enterContext();
        try {
            Object result = cx.evaluateString(scope, code, "agent.js", 1, null);
            String resultStr = org.mozilla.javascript.Context.toString(result);
            if (actionLog.length() > 0) {
                actionLog.append("\n[Script returned] ").append(resultStr);
                return actionLog.toString();
            }
            return resultStr;
        } catch (Throwable e) {
            String error;
            if (Thread.interrupted()) {
                error = "[JS Error] Script execution timeout (" + TIMEOUT_MS / 1000 + "s)";
            } else {
                error = "[JS Error] " + e.getClass().getSimpleName() + ": " + e.getMessage();
            }
            if (actionLog.length() > 0) {
                actionLog.append("\n").append(error);
                return actionLog.toString();
            }
            return error;
        } finally {
            timeoutFuture.cancel(false);
            watchdog.shutdown();
            // Clear interrupted status so it doesn't leak to caller
            Thread.interrupted();
            org.mozilla.javascript.Context.exit();
        }
    }

    private void checkTimeout() {
        if (Thread.currentThread().isInterrupted()) {
            throw new RuntimeException("Script execution timeout (" + TIMEOUT_MS / 1000 + "s)");
        }
    }

    private void appendLog(String entry) {
        if (actionLog.length() > 0) {
            actionLog.append("\n");
        }
        actionLog.append(entry);
    }

    private String saveScreen(String tree) {
        screenCounter++;
        String filename = String.format(Locale.US, "screen_%03d.txt", screenCounter);
        try {
            if (!screensDir.exists()) screensDir.mkdirs();
            File file = new File(screensDir, filename);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(tree);
            writer.flush();
            writer.close();
        } catch (Exception ignored) {
        }
        return filename;
    }

    private void registerFunctions(org.mozilla.javascript.Context cx) {
        // get_screen()
        scope.put("get_screen", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                getScreenCount++;
                if (getScreenCount > MAX_GET_SCREEN_PER_EXEC) {
                    throw new RuntimeException(
                            "get_screen() called " + getScreenCount + " times in one execute_js. " +
                            "Max is " + MAX_GET_SCREEN_PER_EXEC + ". " +
                            "Return result and plan next actions in a new execute_js call.");
                }
                String tree = service.getAccessibilityTree();
                String filename = saveScreen(tree);
                appendLog("[get_screen] saved " + filename + " (" + tree.length() + " chars)");
                return tree;
            }
        });

        // click(target)
        scope.put("click", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) {
                    appendLog("[click] Error: no argument");
                    return false;
                }
                return handleClick(args[0], false);
            }
        });

        // long_click(target)
        scope.put("long_click", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) {
                    appendLog("[long_click] Error: no argument");
                    return false;
                }
                return handleClick(args[0], true);
            }
        });

        // scroll(direction)
        scope.put("scroll", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) {
                    appendLog("[scroll] Error: no argument");
                    return false;
                }
                String direction = org.mozilla.javascript.Context.toString(args[0]);
                boolean result = service.scrollScreen(direction);
                appendLog("[scroll] " + direction + " -> " + result);
                return result;
            }
        });

        // type_text(text)
        scope.put("type_text", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) {
                    appendLog("[type_text] Error: no argument");
                    return false;
                }
                String text = org.mozilla.javascript.Context.toString(args[0]);
                boolean result = service.inputText(text);
                appendLog("[type_text] \"" + text + "\" -> " + result);
                return result;
            }
        });

        // press_home()
        scope.put("press_home", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                boolean result = service.globalAction(AccessibilityService.GLOBAL_ACTION_HOME);
                appendLog("[press_home] -> " + result);
                return result;
            }
        });

        // press_back()
        scope.put("press_back", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                boolean result = service.globalAction(AccessibilityService.GLOBAL_ACTION_BACK);
                appendLog("[press_back] -> " + result);
                return result;
            }
        });

        // press_recents()
        scope.put("press_recents", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                boolean result = service.globalAction(AccessibilityService.GLOBAL_ACTION_RECENTS);
                appendLog("[press_recents] -> " + result);
                return result;
            }
        });

        // show_notifications()
        scope.put("show_notifications", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                boolean result = service.globalAction(AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
                appendLog("[show_notifications] -> " + result);
                return result;
            }
        });

        // sleep(ms)
        scope.put("sleep", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) return null;
                int ms = (int) org.mozilla.javascript.Context.toNumber(args[0]);
                try {
                    Thread.sleep(ms);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Script execution timeout (" + TIMEOUT_MS / 1000 + "s)");
                }
                appendLog("[sleep] " + ms + "ms");
                return null;
            }
        });

        // log(message)
        scope.put("log", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                if (args.length < 1) return null;
                String msg = org.mozilla.javascript.Context.toString(args[0]);
                appendLog("[log] " + msg);
                Log.d(TAG, "[JS log] " + msg);
                return null;
            }
        });

        // list_apps()
        scope.put("list_apps", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                String result = service.listApps();
                appendLog("[list_apps] returned " + result.split("\n").length + " apps");
                return result;
            }
        });

        // launch_app(name)
        scope.put("launch_app", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 1) {
                    appendLog("[launch_app] Error: no argument");
                    return "Error: no app name provided";
                }
                String name = org.mozilla.javascript.Context.toString(args[0]);
                String result = service.launchApp(name);
                appendLog("[launch_app] \"" + name + "\" -> " + result);
                return result;
            }
        });

        // scroll_element(text, direction)
        scope.put("scroll_element", scope, new BaseFunction() {
            @Override
            public Object call(org.mozilla.javascript.Context cx, Scriptable scope,
                               Scriptable thisObj, Object[] args) {
                checkTimeout();
                if (args.length < 2) {
                    appendLog("[scroll_element] Error: need text and direction");
                    return "Error: need text and direction arguments";
                }
                String text = org.mozilla.javascript.Context.toString(args[0]);
                String direction = org.mozilla.javascript.Context.toString(args[1]);
                String result = service.scrollElementByText(text, direction);
                appendLog("[scroll_element] \"" + text + "\" " + direction + " -> " + result);
                return result;
            }
        });
    }

    private Object handleClick(Object arg, boolean longPress) {
        String actionName = longPress ? "long_click" : "click";

        if (arg instanceof String || arg instanceof CharSequence) {
            // click("text") - match text or desc
            String text = arg.toString();
            boolean result;
            if (longPress) {
                result = service.longClickByText(text);
            } else {
                result = service.clickByText(text);
            }
            appendLog("[" + actionName + "] text=\"" + text + "\" -> " + result);
            return result;
        }

        if (arg instanceof NativeObject) {
            NativeObject obj = (NativeObject) arg;

            // click({desc: "X"}) - match desc only
            Object descVal = obj.get("desc", obj);
            if (descVal != null && descVal != Scriptable.NOT_FOUND) {
                String desc = org.mozilla.javascript.Context.toString(descVal);
                boolean result;
                if (longPress) {
                    result = service.longClickByDesc(desc);
                } else {
                    result = service.clickByDesc(desc);
                }
                appendLog("[" + actionName + "] desc=\"" + desc + "\" -> " + result);
                return result;
            }

            // click({x: N, y: N}) - coordinates
            Object xVal = obj.get("x", obj);
            Object yVal = obj.get("y", obj);
            if (xVal != null && xVal != Scriptable.NOT_FOUND
                    && yVal != null && yVal != Scriptable.NOT_FOUND) {
                int x = (int) org.mozilla.javascript.Context.toNumber(xVal);
                int y = (int) org.mozilla.javascript.Context.toNumber(yVal);
                boolean result;
                if (longPress) {
                    result = service.longClickByCoordinates(x, y);
                } else {
                    result = service.clickByCoordinates(x, y);
                }
                appendLog("[" + actionName + "] coords=(" + x + "," + y + ") -> " + result);
                return result;
            }

            appendLog("[" + actionName + "] Error: object must have 'desc' or 'x'+'y' properties");
            return false;
        }

        appendLog("[" + actionName + "] Error: invalid argument type");
        return false;
    }
}
