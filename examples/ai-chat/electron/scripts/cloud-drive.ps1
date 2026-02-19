# cloud-drive.ps1 — Register a CloudFS PSDrive backed by HTTP /api/files/* endpoints.
# Requires $env:CLOUD_BASE to be set (e.g. "https://ai.connect-screen.com").
# Returns immediately if $env:CLOUD_BASE is not set.

if (-not $env:CLOUD_BASE) { return }

$dllVersion = "v4"
$dllPath = Join-Path $env:TEMP "CloudFSProvider-$dllVersion.dll"

if (-not (Test-Path $dllPath)) {
    $smaAssembly = [System.Management.Automation.PSObject].Assembly.Location
    Add-Type -OutputAssembly $dllPath -ReferencedAssemblies $smaAssembly -TypeDefinition @"
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Management.Automation;
using System.Management.Automation.Provider;
using System.Net;
using System.Text;

[CmdletProvider("CloudFS", ProviderCapabilities.None)]
public class CloudFSProvider : NavigationCmdletProvider, IContentCmdletProvider
{
    private static string _baseUrl = "";

    public static void SetBaseUrl(string url)
    {
        // Strip trailing slash
        if (url != null && url.EndsWith("/"))
        {
            url = url.Substring(0, url.Length - 1);
        }
        _baseUrl = url ?? "";
    }

    // --- Path helpers ---

    private static string NormalizePath(string path)
    {
        if (path == null) return "/";
        path = path.Replace("\\", "/");
        // Remove drive prefix like "cloud:" if present
        if (path.Length >= 2 && path[1] == ':')
        {
            path = path.Substring(2);
        }
        if (!path.StartsWith("/"))
        {
            path = "/" + path;
        }
        // Collapse double slashes
        while (path.Contains("//"))
        {
            path = path.Replace("//", "/");
        }
        // Remove trailing slash (except for root)
        if (path.Length > 1 && path.EndsWith("/"))
        {
            path = path.Substring(0, path.Length - 1);
        }
        return path;
    }

    protected override string MakePath(string parent, string child)
    {
        string p = NormalizePath(parent);
        if (string.IsNullOrEmpty(child)) return p;
        child = child.Replace("\\", "/").TrimStart('/');
        if (p == "/") return "/" + child;
        return p + "/" + child;
    }

    protected override string GetChildName(string path)
    {
        string p = NormalizePath(path);
        if (p == "/") return "";
        int idx = p.LastIndexOf('/');
        if (idx < 0) return p;
        return p.Substring(idx + 1);
    }

    protected override string GetParentPath(string path, string root)
    {
        string p = NormalizePath(path);
        if (p == "/" || p == "") return "";
        int idx = p.LastIndexOf('/');
        if (idx <= 0) return "/";
        return p.Substring(0, idx);
    }

    protected override bool IsValidPath(string path)
    {
        return true;
    }

    // --- HTTP helpers ---

    private WebClient CreateWebClient()
    {
        WebClient wc = new WebClient();
        wc.Encoding = Encoding.UTF8;
        string cookie = Environment.GetEnvironmentVariable("CLOUD_COOKIE");
        if (!string.IsNullOrEmpty(cookie))
        {
            wc.Headers[HttpRequestHeader.Cookie] = "session=" + cookie;
        }
        return wc;
    }

    private string HttpGet(string relativeUrl)
    {
        using (WebClient wc = CreateWebClient())
        {
            return wc.DownloadString(_baseUrl + relativeUrl);
        }
    }

    private string HttpPut(string relativeUrl, string body)
    {
        using (WebClient wc = CreateWebClient())
        {
            wc.Headers[HttpRequestHeader.ContentType] = "application/octet-stream";
            return wc.UploadString(_baseUrl + relativeUrl, "PUT", body);
        }
    }

    private string HttpPost(string relativeUrl, string body)
    {
        using (WebClient wc = CreateWebClient())
        {
            wc.Headers[HttpRequestHeader.ContentType] = "application/json";
            return wc.UploadString(_baseUrl + relativeUrl, "POST", body);
        }
    }

    private string HttpDelete(string relativeUrl)
    {
        using (WebClient wc = CreateWebClient())
        {
            return wc.UploadString(_baseUrl + relativeUrl, "DELETE", "");
        }
    }

    // JSON parsing via PowerShell's ConvertFrom-Json
    private PSObject ParseJson(string json)
    {
        // Base64-encode the JSON to avoid quoting issues
        string b64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(json));
        string script = "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + b64 + "')) | ConvertFrom-Json";
        Collection<PSObject> results = this.InvokeCommand.InvokeScript(script);
        if (results != null && results.Count > 0)
        {
            return results[0];
        }
        return null;
    }

    private static string UrlEncode(string value)
    {
        // Manual URL encoding for compatibility
        if (value == null) return "";
        StringBuilder sb = new StringBuilder();
        foreach (char c in value)
        {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
                || c == '-' || c == '_' || c == '.' || c == '~' || c == '/')
            {
                sb.Append(c);
            }
            else
            {
                byte[] bytes = Encoding.UTF8.GetBytes(new char[] { c });
                foreach (byte b in bytes)
                {
                    sb.Append('%');
                    sb.Append(b.ToString("X2"));
                }
            }
        }
        return sb.ToString();
    }

    // --- Stat helper ---

    private PSObject StatPath(string path)
    {
        string p = NormalizePath(path);
        try
        {
            string json = HttpGet("/api/files/stat?path=" + UrlEncode(p));
            return ParseJson(json);
        }
        catch (WebException)
        {
            return null;
        }
    }

    // --- ItemCmdletProvider ---

    protected override bool ItemExists(string path)
    {
        string p = NormalizePath(path);
        if (p == "/") return true;
        return StatPath(p) != null;
    }

    protected override void GetItem(string path)
    {
        string p = NormalizePath(path);
        PSObject stat = StatPath(p);
        if (stat == null)
        {
            WriteError(new ErrorRecord(
                new ItemNotFoundException("Path not found: " + p),
                "PathNotFound", ErrorCategory.ObjectNotFound, p));
            return;
        }

        PSObject item = new PSObject();
        item.Properties.Add(new PSNoteProperty("PSPath", "CloudFS::" + p));
        item.Properties.Add(new PSNoteProperty("Name", GetChildName(p)));

        // Copy properties from stat
        foreach (PSPropertyInfo prop in stat.Properties)
        {
            item.Properties.Add(new PSNoteProperty(prop.Name, prop.Value));
        }

        WriteItemObject(item, p, IsItemContainer(path));
    }

    protected override bool IsItemContainer(string path)
    {
        string p = NormalizePath(path);
        if (p == "/") return true;
        PSObject stat = StatPath(p);
        if (stat == null) return false;
        PSPropertyInfo isDirProp = stat.Properties["isDirectory"];
        if (isDirProp != null && isDirProp.Value != null)
        {
            return Convert.ToBoolean(isDirProp.Value);
        }
        return false;
    }

    // --- ContainerCmdletProvider ---

    protected override void GetChildItems(string path, bool recurse)
    {
        string p = NormalizePath(path);
        try
        {
            string listUrl = "/api/files/list?path=" + UrlEncode(p);
            if (recurse)
            {
                listUrl += "&recursive=1";
            }
            string json = HttpGet(listUrl);
            PSObject parsed = ParseJson(json);
            if (parsed == null) return;

            // The response has an "entries" array
            PSPropertyInfo entriesProp = parsed.Properties["entries"];
            if (entriesProp == null || entriesProp.Value == null) return;

            object[] entries;
            if (entriesProp.Value is object[])
            {
                entries = (object[])entriesProp.Value;
            }
            else if (entriesProp.Value is PSObject)
            {
                // Single item
                entries = new object[] { entriesProp.Value };
            }
            else
            {
                return;
            }

            foreach (object entry in entries)
            {
                PSObject entryObj = entry as PSObject;
                if (entryObj == null) continue;

                PSPropertyInfo nameProp = entryObj.Properties["name"];
                if (nameProp == null) continue;
                string name = nameProp.Value.ToString();

                // Use "path" field if present (recursive response), otherwise compute from name
                string childPath;
                PSPropertyInfo pathProp = entryObj.Properties["path"];
                if (pathProp != null && pathProp.Value != null)
                {
                    childPath = MakePath(p, pathProp.Value.ToString());
                }
                else
                {
                    childPath = MakePath(p, name);
                }

                bool isDir = false;
                PSPropertyInfo isDirProp = entryObj.Properties["isDirectory"];
                if (isDirProp != null && isDirProp.Value != null)
                {
                    isDir = Convert.ToBoolean(isDirProp.Value);
                }

                PSObject item = new PSObject();
                item.Properties.Add(new PSNoteProperty("PSPath", "CloudFS::" + childPath));
                item.Properties.Add(new PSNoteProperty("Name", name));
                foreach (PSPropertyInfo prop in entryObj.Properties)
                {
                    if (prop.Name != "Name")
                    {
                        item.Properties.Add(new PSNoteProperty(prop.Name, prop.Value));
                    }
                }

                WriteItemObject(item, childPath, isDir);
            }
        }
        catch (WebException ex)
        {
            WriteError(new ErrorRecord(ex, "ListFailed", ErrorCategory.ReadError, path));
        }
    }

    protected override bool HasChildItems(string path)
    {
        return IsItemContainer(path);
    }

    protected override void NewItem(string path, string itemTypeName, object newItemValue)
    {
        string p = NormalizePath(path);
        try
        {
            if (string.Equals(itemTypeName, "Directory", StringComparison.OrdinalIgnoreCase))
            {
                HttpPost("/api/files/mkdir", "{\"path\":\"" + p.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"}");
                PSObject item = new PSObject();
                item.Properties.Add(new PSNoteProperty("PSPath", "CloudFS::" + p));
                item.Properties.Add(new PSNoteProperty("Name", GetChildName(p)));
                item.Properties.Add(new PSNoteProperty("isDirectory", true));
                WriteItemObject(item, p, true);
            }
            else
            {
                // File — write content if provided
                string content = newItemValue != null ? newItemValue.ToString() : "";
                HttpPut("/api/files/content?path=" + UrlEncode(p), content);
                PSObject item = new PSObject();
                item.Properties.Add(new PSNoteProperty("PSPath", "CloudFS::" + p));
                item.Properties.Add(new PSNoteProperty("Name", GetChildName(p)));
                item.Properties.Add(new PSNoteProperty("isFile", true));
                WriteItemObject(item, p, false);
            }
        }
        catch (WebException ex)
        {
            WriteError(new ErrorRecord(ex, "NewItemFailed", ErrorCategory.WriteError, path));
        }
    }

    protected override void RemoveItem(string path, bool recurse)
    {
        string p = NormalizePath(path);
        try
        {
            string recursive = recurse ? "1" : "0";
            HttpDelete("/api/files?path=" + UrlEncode(p) + "&recursive=" + recursive);
        }
        catch (WebException ex)
        {
            WriteError(new ErrorRecord(ex, "RemoveFailed", ErrorCategory.WriteError, path));
        }
    }

    // --- IContentCmdletProvider ---

    public IContentReader GetContentReader(string path)
    {
        string p = NormalizePath(path);
        try
        {
            string content = HttpGet("/api/files/content?path=" + UrlEncode(p));
            return new CloudContentReader(content);
        }
        catch (WebException ex)
        {
            WriteError(new ErrorRecord(ex, "ReadFailed", ErrorCategory.ReadError, path));
            return null;
        }
    }

    public object GetContentReaderDynamicParameters(string path) { return null; }

    public IContentWriter GetContentWriter(string path)
    {
        string p = NormalizePath(path);
        return new CloudContentWriter(this, p);
    }

    public object GetContentWriterDynamicParameters(string path) { return null; }

    public void ClearContent(string path)
    {
        string p = NormalizePath(path);
        try
        {
            HttpPut("/api/files/content?path=" + UrlEncode(p), "");
        }
        catch (WebException ex)
        {
            WriteError(new ErrorRecord(ex, "ClearFailed", ErrorCategory.WriteError, path));
        }
    }

    public object ClearContentDynamicParameters(string path) { return null; }

    // --- Content reader ---

    internal class CloudContentReader : IContentReader
    {
        private string[] _lines;
        private int _index;

        public CloudContentReader(string content)
        {
            _lines = (content ?? "").Split(new string[] { "\r\n", "\n" }, StringSplitOptions.None);
            _index = 0;
        }

        public IList Read(long readCount)
        {
            ArrayList result = new ArrayList();
            long count = readCount;
            while (_index < _lines.Length && count > 0)
            {
                result.Add(_lines[_index]);
                _index++;
                count--;
            }
            return result;
        }

        public void Seek(long offset, SeekOrigin origin)
        {
            if (origin == SeekOrigin.Begin)
            {
                _index = (int)offset;
            }
        }

        public void Close() { }
        public void Dispose() { }
    }

    // --- Content writer ---

    internal class CloudContentWriter : IContentWriter
    {
        private CloudFSProvider _provider;
        private string _path;
        private List<string> _lines;

        public CloudContentWriter(CloudFSProvider provider, string path)
        {
            _provider = provider;
            _path = path;
            _lines = new List<string>();
        }

        public IList Write(IList content)
        {
            foreach (object item in content)
            {
                _lines.Add(item != null ? item.ToString() : "");
            }
            return content;
        }

        public void Seek(long offset, SeekOrigin origin) { }

        public void Close()
        {
            try
            {
                string body = string.Join("\n", _lines.ToArray());
                _provider.HttpPut("/api/files/content?path=" + UrlEncode(_path), body);
            }
            catch (Exception)
            {
                // Swallow — errors already reported via WriteError in calling context
            }
        }

        public void Dispose() { }
    }
}
"@
}

Import-Module $dllPath -ErrorAction Stop

[CloudFSProvider]::SetBaseUrl($env:CLOUD_BASE)

# Remove existing drive if present (re-entrant calls)
if (Get-PSDrive -Name cloud -ErrorAction SilentlyContinue) {
    Remove-PSDrive -Name cloud -Force -ErrorAction SilentlyContinue
}

New-PSDrive -Name cloud -PSProvider CloudFS -Root "/" -Scope Global | Out-Null
