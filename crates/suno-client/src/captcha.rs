//! hCaptcha bypass via piloted Chrome.
//!
//! Suno gates `/api/generate/v2-web/` with an invisible hCaptcha challenge.
//! The request body's `token` field must contain a freshly-solved hCaptcha
//! response. Headless HTTP clients can't pass it; only a real browser with
//! a warm behavioural fingerprint does.
//!
//! This module spawns a hidden Chrome instance with `--remote-debugging-port`
//! enabled, injects the user's Suno cookies via CDP, navigates to
//! suno.com/create, then renders an invisible hCaptcha widget and calls
//! `hcaptcha.execute()` to obtain a token. The Chrome instance is reused
//! across calls so subsequent generations are fast.
//!
//! Discovered + verified end-to-end on 2026-04-08.

use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message;

use crate::auth::AuthState;
use crate::error::SunoError;

/// Suno's hCaptcha sitekey, captured from the live web app's
/// `hcaptcha.render(...)` arguments on 2026-04-08.
const SUNO_HCAPTCHA_SITEKEY: &str = "d65453de-3f1a-4aac-9366-a0f06e52b2ce";
/// Port for the suno-cli managed Chrome instance. Picked high to avoid
/// colliding with the user's main Chrome (which is rarely on 9233).
const CDP_PORT: u16 = 9233;
const CDP_HOST: &str = "127.0.0.1";

/// Singleton holder for the Chrome process so the same instance survives
/// across multiple `generate()` calls within one CLI invocation.
static CHROME: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

fn chrome_slot() -> &'static Mutex<Option<Child>> {
    CHROME.get_or_init(|| Mutex::new(None))
}

/// Solve a fresh hCaptcha challenge and return the token to attach to a
/// `/api/generate/v2-web/` request body.
pub async fn solve(auth: &AuthState) -> Result<String, SunoError> {
    ensure_chrome_running().await?;
    let target = find_or_create_suno_tab().await?;
    let token = render_and_execute(&target.web_socket_debugger_url, auth).await?;
    Ok(token)
}

/// Either reuse a Chrome instance already listening on `CDP_PORT` or spawn
/// a new hidden one with the suno-cli profile dir. Idempotent.
async fn ensure_chrome_running() -> Result<(), SunoError> {
    if cdp_version().await.is_ok() {
        return Ok(());
    }

    // Need to spawn it.
    let chrome_path = locate_chrome()?;
    // Vendored from upstream `directories::ProjectDirs("com", "suno-cli", "suno-cli")`
    // → switched to `dirs::data_local_dir()` so we depend on one less crate. The
    // resulting path differs from upstream (e.g.
    // `~/Library/Application Support/melodie/chrome-profile` on macOS) which is
    // intentional: we want a separate profile from any local `suno-cli`
    // install.
    let profile_dir = dirs::data_local_dir()
        .map(|d| d.join("melodie").join("chrome-profile"))
        .ok_or_else(|| SunoError::Config("could not resolve data dir for chrome profile".into()))?;
    std::fs::create_dir_all(&profile_dir)?;

    eprintln!("Launching headless Chrome for captcha solver (one-time per session)...");

    // NOTE: do NOT use --headless. hCaptcha's bot-detection trips on headless
    // mode and returns "challenge-expired". We run a real headed Chrome but
    // shove it far offscreen + give it a 1x1 window so the user never sees it.
    let child = Command::new(&chrome_path)
        .arg(format!("--remote-debugging-port={CDP_PORT}"))
        .arg(format!("--user-data-dir={}", profile_dir.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("--disable-search-engine-choice-screen")
        .arg("--disable-features=TranslateUI")
        .arg("--window-position=-32000,-32000")
        .arg("--window-size=1,1")
        .arg("--silent-launch")
        .arg("about:blank")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| SunoError::Config(format!("failed to spawn Chrome at {chrome_path:?}: {e}")))?;

    {
        let mut slot = chrome_slot().lock().await;
        *slot = Some(child);
    }

    // Wait up to 10s for CDP to come up
    for _ in 0..20 {
        sleep(Duration::from_millis(500)).await;
        if cdp_version().await.is_ok() {
            return Ok(());
        }
    }

    Err(SunoError::Config(
        "Chrome was spawned but never opened the CDP port. Try `suno chrome-launch` for a visible Chrome window.".into(),
    ))
}

/// Locate a Chrome binary on the host. Looks in the usual macOS / Linux /
/// Windows install paths and falls back to `$PATH`.
fn locate_chrome() -> Result<String, SunoError> {
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        &[
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    } else if cfg!(target_os = "linux") {
        &[
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
    } else {
        &[
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        ]
    };
    for c in candidates {
        if std::path::Path::new(c).exists() {
            return Ok(c.to_string());
        }
    }
    Err(SunoError::Config(
        "Could not find a Chrome/Chromium binary. Install Google Chrome or set SUNO_CHROME_PATH."
            .into(),
    ))
}

/// CDP target descriptor returned by `/json/list`.
#[derive(Debug, Deserialize)]
struct Target {
    #[serde(rename = "type")]
    target_type: String,
    url: String,
    #[serde(rename = "webSocketDebuggerUrl")]
    web_socket_debugger_url: String,
}

async fn cdp_version() -> Result<serde_json::Value, SunoError> {
    let url = format!("http://{CDP_HOST}:{CDP_PORT}/json/version");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map_err(|e| SunoError::Config(format!("CDP /json/version: {e}")))?;
    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| SunoError::Config(format!("CDP json parse: {e}")))?;
    Ok(v)
}

async fn cdp_list() -> Result<Vec<Target>, SunoError> {
    let url = format!("http://{CDP_HOST}:{CDP_PORT}/json/list");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| SunoError::Config(format!("CDP /json/list: {e}")))?;
    let list: Vec<Target> = resp
        .json()
        .await
        .map_err(|e| SunoError::Config(format!("CDP json parse: {e}")))?;
    Ok(list)
}

/// Find the existing suno.com tab in the managed Chrome, or open a new one
/// at suno.com/create.
async fn find_or_create_suno_tab() -> Result<Target, SunoError> {
    let targets = cdp_list().await?;
    if let Some(t) = targets
        .into_iter()
        .find(|t| t.target_type == "page" && t.url.contains("suno.com"))
    {
        return Ok(t);
    }

    // No suno tab — open one. CDP exposes a /json/new?url= helper.
    let url = format!(
        "http://{CDP_HOST}:{CDP_PORT}/json/new?{}",
        urlencode("https://suno.com/create")
    );
    let resp = reqwest::Client::new()
        .put(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| SunoError::Config(format!("CDP /json/new: {e}")))?;
    let t: Target = resp
        .json()
        .await
        .map_err(|e| SunoError::Config(format!("CDP /json/new parse: {e}")))?;
    // Give the page a moment to start loading before we attach
    sleep(Duration::from_millis(800)).await;
    Ok(t)
}

fn urlencode(s: &str) -> String {
    s.replace(":", "%3A").replace("/", "%2F")
}

/// CDP request envelope.
#[derive(Serialize)]
struct CdpReq<'a> {
    id: u64,
    method: &'a str,
    params: serde_json::Value,
}

/// CDP cookie struct (subset of Network.CookieParam).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CdpCookie {
    name: String,
    value: String,
    domain: String,
    path: String,
    secure: bool,
    http_only: bool,
    same_site: &'static str,
}

type CdpStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Send a CDP method and wait for the response (drains intervening events).
async fn cdp_call(
    ws: &mut CdpStream,
    id: u64,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, SunoError> {
    let req = CdpReq { id, method, params };
    let payload = serde_json::to_string(&req).unwrap();
    ws.send(Message::Text(payload.into()))
        .await
        .map_err(|e| SunoError::Config(format!("CDP ws send {method}: {e}")))?;
    loop {
        let msg = timeout(Duration::from_secs(60), ws.next())
            .await
            .map_err(|_| SunoError::Config(format!("CDP {method} timeout")))?
            .ok_or_else(|| SunoError::Config(format!("CDP {method} ws closed")))?
            .map_err(|e| SunoError::Config(format!("CDP {method} ws err: {e}")))?;
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {
                continue;
            }
            Message::Close(_) => {
                return Err(SunoError::Config(format!("CDP {method} ws closed mid-call")));
            }
        };
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| SunoError::Config(format!("CDP {method} json: {e}")))?;
        if v.get("id").and_then(|x| x.as_u64()) == Some(id) {
            if let Some(err) = v.get("error") {
                return Err(SunoError::Config(format!("CDP {method} error: {err}")));
            }
            return Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null));
        }
        // Ignore unrelated events
    }
}

/// Connect to the page websocket, inject cookies, navigate to suno.com/create
/// if needed, then render an invisible hCaptcha widget and call
/// `hcaptcha.execute()` to obtain a token.
async fn render_and_execute(ws_url: &str, _auth: &AuthState) -> Result<String, SunoError> {
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_url)
        .await
        .map_err(|e| SunoError::Config(format!("CDP ws connect: {e}")))?;

    let mut next_id: u64 = 0;
    let mut next = || -> u64 {
        next_id += 1;
        next_id
    };

    cdp_call(&mut ws, next(), "Network.enable", serde_json::json!({})).await?;
    cdp_call(&mut ws, next(), "Page.enable", serde_json::json!({})).await?;
    cdp_call(&mut ws, next(), "Runtime.enable", serde_json::json!({})).await?;

    // Inject cookies fresh from rookie every time so we always have the
    // latest from the user's main Chrome.
    let cookies = extract_cookies()?;
    if !cookies.is_empty() {
        cdp_call(
            &mut ws,
            next(),
            "Network.setCookies",
            serde_json::json!({ "cookies": cookies }),
        )
        .await?;
    }

    // Probe current URL — navigate if not already on suno.com/create
    let page_url = cdp_call(
        &mut ws,
        next(),
        "Runtime.evaluate",
        serde_json::json!({
            "expression": "location.href",
            "returnByValue": true,
        }),
    )
    .await?;
    let needs_nav = page_url
        .get("result")
        .and_then(|r| r.get("value"))
        .and_then(|v| v.as_str())
        .map(|s| !s.contains("suno.com/create"))
        .unwrap_or(true);
    if needs_nav {
        cdp_call(
            &mut ws,
            next(),
            "Page.navigate",
            serde_json::json!({ "url": "https://suno.com/create" }),
        )
        .await?;
        // Poll for hcaptcha global (up to 30s)
        let mut ready = false;
        for _ in 0..30 {
            sleep(Duration::from_secs(1)).await;
            let probe = cdp_call(
                &mut ws,
                next(),
                "Runtime.evaluate",
                serde_json::json!({
                    "expression": "typeof hcaptcha !== 'undefined' && !!hcaptcha.render",
                    "returnByValue": true,
                }),
            )
            .await?;
            if probe
                .get("result")
                .and_then(|r| r.get("value"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                ready = true;
                break;
            }
        }
        if !ready {
            return Err(SunoError::Config(
                "hcaptcha never finished loading on suno.com/create".into(),
            ));
        }
        // Extra settle so the SDK is fully wired up
        sleep(Duration::from_secs(2)).await;
    }

    // Render an invisible widget and execute it
    let solve_js = format!(
        r#"
        (async () => {{
            try {{
                const div = document.createElement('div');
                div.style.cssText = 'position:fixed;top:-9999px;left:-9999px;';
                document.body.appendChild(div);
                const id = hcaptcha.render(div, {{
                    sitekey: '{SUNO_HCAPTCHA_SITEKEY}',
                    size: 'invisible',
                    sentry: false,
                    endpoint: 'https://hcaptcha-endpoint-prod.suno.com',
                    assethost: 'https://hcaptcha-assets-prod.suno.com',
                    imghost: 'https://hcaptcha-imgs-prod.suno.com',
                    reportapi: 'https://hcaptcha-reportapi-prod.suno.com',
                }});
                const r = await hcaptcha.execute(id, {{ async: true }});
                return (r && r.response) ? r.response : '';
            }} catch (e) {{
                return 'ERR:' + String(e);
            }}
        }})()
        "#
    );

    let result = cdp_call(
        &mut ws,
        next(),
        "Runtime.evaluate",
        serde_json::json!({
            "expression": solve_js,
            "awaitPromise": true,
            "returnByValue": true,
        }),
    )
    .await?;

    let token = result
        .get("result")
        .and_then(|r| r.get("value"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if token.is_empty() {
        return Err(SunoError::Config("hcaptcha returned empty token".into()));
    }
    if token.starts_with("ERR:") {
        return Err(SunoError::Config(format!("hcaptcha solver: {token}")));
    }
    Ok(token)
}

/// Pull the user's Suno cookies from their main Chrome via `rookie`.
/// Returns them in CDP `Network.CookieParam` shape.
fn extract_cookies() -> Result<Vec<CdpCookie>, SunoError> {
    let domains: Vec<String> = vec![
        "suno.com".into(),
        "auth.suno.com".into(),
        ".suno.com".into(),
    ];
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let raw_cookies = match rookie::chrome(Some(domains)) {
        Ok(cs) => cs,
        Err(e) => {
            return Err(SunoError::Config(format!(
                "could not read Chrome cookies via rookie: {e}"
            )));
        }
    };

    for c in raw_cookies {
        if !c.domain.contains("suno.com") {
            continue;
        }
        let key = (c.name.clone(), c.domain.clone());
        if !seen.insert(key) {
            continue;
        }
        out.push(CdpCookie {
            name: c.name,
            value: c.value,
            domain: c.domain,
            path: c.path,
            secure: c.secure,
            http_only: c.http_only,
            same_site: "Lax",
        });
    }
    Ok(out)
}

/// Drain the spawned Chrome's stderr in the background — keeps it from
/// blocking on a full pipe and gives us logs for debugging.
#[allow(dead_code)]
fn drain_stderr(child: &mut Child) {
    if let Some(stderr) = child.stderr.take() {
        let mut reader = BufReader::new(stderr).lines();
        tokio::spawn(async move {
            while let Ok(Some(_)) = reader.next_line().await {
                // discard
            }
        });
    }
}
