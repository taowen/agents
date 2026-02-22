package ai.connct_screen.rn;

import android.content.Context;
import android.util.Log;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.Locale;

/**
 * Java-side callbacks for the "app" agent type (Android accessibility / app automation).
 * Each static method is called from C++ (tools_app.cpp) via JNI.
 */
public class AppToolsHost {

    private static final String TAG = "AppToolsHost";
    private static int screenCounter = 0;

    public static String nativeTakeScreenshot() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "ERROR: accessibility service not running";
        return service.takeScreenshotSync();
    }

    public static String nativeGetScreen() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "(accessibility service not running)";
        String tree = service.getAccessibilityTree();
        screenCounter++;
        // Save screen dump to file for debugging
        try {
            Context ctx = service.getApplicationContext();
            File screensDir = new File(ctx.getFilesDir(), "screens");
            if (!screensDir.exists()) screensDir.mkdirs();
            String filename = String.format(Locale.US, "screen_%03d.txt", screenCounter);
            File file = new File(screensDir, filename);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(tree);
            writer.flush();
            writer.close();
            Log.d(TAG, "[getScreen] saved " + filename + " (" + tree.length() + " chars)");
        } catch (Exception ignored) {}
        return tree;
    }

    public static boolean nativeClickByText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByText(text);
    }

    public static boolean nativeClickByDesc(String desc) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByDesc(desc);
    }

    public static boolean nativeClickByCoords(int x, int y) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByCoordinates(x, y);
    }

    public static boolean nativeLongClickByText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByText(text);
    }

    public static boolean nativeLongClickByDesc(String desc) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByDesc(desc);
    }

    public static boolean nativeLongClickByCoords(int x, int y) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByCoordinates(x, y);
    }

    public static boolean nativeScrollScreen(String direction) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.scrollScreen(direction);
    }

    public static String nativeScrollElement(String text, String direction) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.scrollElementByText(text, direction);
    }

    public static boolean nativeTypeText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.inputText(text);
    }

    public static boolean nativePressHome() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME);
    }

    public static boolean nativePressBack() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK);
    }

    public static boolean nativePressRecents() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_RECENTS);
    }

    public static boolean nativeShowNotifications() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
    }

    public static String nativeLaunchApp(String name) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.launchApp(name);
    }

    public static String nativeListApps() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.listApps();
    }
}
