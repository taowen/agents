package com.google.android.accessibility.selecttospeak;

import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.util.Log;

import java.util.List;

/**
 * App listing and launching via PackageManager.
 * Extracted from SelectToSpeakService.
 */
public class AppLauncher {

    private static final String TAG = "A11yAgent";

    private final Context context;

    public AppLauncher(Context context) {
        this.context = context;
    }

    public String listApps() {
        PackageManager pm = context.getPackageManager();
        List<ApplicationInfo> apps = pm.getInstalledApplications(0);
        StringBuilder sb = new StringBuilder();
        for (ApplicationInfo app : apps) {
            Intent launchIntent = pm.getLaunchIntentForPackage(app.packageName);
            if (launchIntent != null) {
                String label = pm.getApplicationLabel(app).toString();
                sb.append(label).append(" (").append(app.packageName).append(")\n");
            }
        }
        return sb.toString().trim();
    }

    public String launchApp(String name) {
        PackageManager pm = context.getPackageManager();

        Intent intent = pm.getLaunchIntentForPackage(name);
        if (intent != null) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
            Log.d(TAG, "[TOOL] launchApp: launched by package " + name);
            return "Launched " + name;
        }

        List<ApplicationInfo> apps = pm.getInstalledApplications(0);
        for (ApplicationInfo app : apps) {
            String label = pm.getApplicationLabel(app).toString();
            if (label.contains(name) || name.contains(label)) {
                Intent launchIntent = pm.getLaunchIntentForPackage(app.packageName);
                if (launchIntent != null) {
                    launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    context.startActivity(launchIntent);
                    Log.d(TAG, "[TOOL] launchApp: launched " + label + " (" + app.packageName + ")");
                    return "Launched " + label + " (" + app.packageName + ")";
                }
            }
        }

        Log.d(TAG, "[TOOL] launchApp: no app found for \"" + name + "\"");
        return "App not found: " + name;
    }
}
