package in.chillar.ml;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Telephony;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Chillar ML for Android: shows the live web app and adds one native power, reading UPI debit SMS on the phone.
 * The bridge only answers pages from the Chillar site; every other link opens in the normal browser.
 */
public class MainActivity extends Activity {
    static final String HOME = "https://ml-project-wheat.vercel.app/";
    static final String HOST = Uri.parse(HOME).getHost();
    private static final int SMS_REQUEST = 7;
    private static final int FILE_REQUEST = 8;

    private WebView web;
    private ValueCallback<Uri[]> pendingFiles;
    private volatile boolean onChillar = false;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(Color.parseColor("#2E5B3C"));
        web = new WebView(this);
        setContentView(web);
        WebSettings ws = web.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true); // the app keeps your data in localStorage
        ws.setAllowFileAccess(false);
        ws.setAllowContentAccess(false);
        web.addJavascriptInterface(new Bridge(), "ChillarAndroid");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("https".equals(u.getScheme()) && HOST.equals(u.getHost())) return false;
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, u));
                } catch (ActivityNotFoundException ignored) {}
                return true;
            }

            @Override
            public void onPageStarted(WebView v, String url, android.graphics.Bitmap icon) {
                Uri u = Uri.parse(url);
                onChillar = "https".equals(u.getScheme()) && HOST.equals(u.getHost());
            }

            @Override
            public void onReceivedError(WebView v, WebResourceRequest req, WebResourceError err) {
                if (!req.isForMainFrame()) return;
                onChillar = false;
                v.loadDataWithBaseURL(null, "<body style='font-family:sans-serif;background:#F6F3EA;color:#1D2A21;padding:40px 24px'>"
                        + "<h2>You're offline</h2><p>Chillar needs the internet to run its models. Connect and tap below.</p>"
                        + "<a href='" + HOME + "' style='display:inline-block;background:#2E5B3C;color:#fff;padding:12px 18px;border-radius:12px;text-decoration:none'>Try again</a></body>",
                        "text/html", "utf-8", null);
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams params) {
                if (pendingFiles != null) pendingFiles.onReceiveValue(null);
                pendingFiles = cb;
                Intent pick = new Intent(Intent.ACTION_GET_CONTENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/pdf");
                try {
                    startActivityForResult(Intent.createChooser(pick, "Choose your PhonePe statement"), FILE_REQUEST);
                } catch (ActivityNotFoundException e) {
                    pendingFiles = null;
                    return false;
                }
                return true;
            }
        });
        if (saved != null) web.restoreState(saved);
        else web.loadUrl(HOME);
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onActivityResult(int code, int result, Intent data) {
        super.onActivityResult(code, result, data);
        if (code != FILE_REQUEST || pendingFiles == null) return;
        Uri u = result == RESULT_OK && data != null ? data.getData() : null;
        pendingFiles.onReceiveValue(u == null ? null : new Uri[]{u});
        pendingFiles = null;
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] grants) {
        if (code != SMS_REQUEST) return;
        boolean ok = grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED;
        web.evaluateJavascript("window.chillarSmsPermission && window.chillarSmsPermission(" + ok + ")", null);
    }

    private boolean smsAllowed() {
        return checkSelfPermission(Manifest.permission.READ_SMS) == PackageManager.PERMISSION_GRANTED;
    }

    /** What the web page can call as window.ChillarAndroid.*. Every method refuses pages that aren't Chillar. */
    final class Bridge {
        @JavascriptInterface
        public boolean hasSmsPermission() {
            return onChillar && smsAllowed();
        }

        @JavascriptInterface
        public void requestSmsPermission() {
            if (!onChillar) return;
            runOnUiThread(() -> requestPermissions(new String[]{Manifest.permission.READ_SMS}, SMS_REQUEST));
        }

        /** UPI/bank debits received since `sinceMillis`, as JSON. Only amount, date, payee and reference leave this method. */
        @JavascriptInterface
        public String readUpiSms(double sinceMillis) {
            JSONArray out = new JSONArray();
            if (!onChillar || !smsAllowed()) return out.toString();
            SimpleDateFormat day = new SimpleDateFormat("yyyy-MM-dd", Locale.US);
            String[] cols = {Telephony.Sms._ID, Telephony.Sms.ADDRESS, Telephony.Sms.BODY, Telephony.Sms.DATE};
            try (Cursor c = getContentResolver().query(Telephony.Sms.Inbox.CONTENT_URI, cols,
                    Telephony.Sms.DATE + " >= ?", new String[]{String.valueOf((long) sinceMillis)}, Telephony.Sms.DATE + " DESC")) {
                if (c == null) return out.toString();
                while (c.moveToNext() && out.length() < 3000) {
                    if (!UpiSms.fromBankSender(c.getString(1))) continue;
                    UpiSms.Debit d = UpiSms.parse(c.getString(2));
                    if (d == null) continue;
                    long when = c.getLong(3);
                    JSONObject o = new JSONObject();
                    o.put("amount", d.amount);
                    o.put("date", day.format(new Date(when)));
                    o.put("ts", when / 1000.0);
                    o.put("payee", d.payee);
                    o.put("ref", "sms:" + (d.ref != null ? d.ref : "id" + c.getLong(0)));
                    out.put(o);
                }
            } catch (SecurityException | JSONException e) {
                return new JSONArray().toString();
            }
            return out.toString();
        }
    }
}
