//! Startup checks that surface missing host requirements before the operator
//! discovers them mid-generation.

use std::path::Path;

const MAC_CANDIDATES: &[&str] = &[
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
];
const LINUX_CANDIDATES: &[&str] = &[
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
];

pub fn check_chrome() {
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        MAC_CANDIDATES
    } else if cfg!(target_os = "linux") {
        LINUX_CANDIDATES
    } else {
        &[]
    };

    let found = candidates.iter().any(|c| Path::new(c).exists());
    if !found {
        tracing::warn!(
            "No Chrome/Chromium binary found in standard install paths. The Suno hCaptcha solver will fail at /api/songs request time. Install Google Chrome or Chromium."
        );
    } else {
        tracing::debug!("Chrome binary detected for hCaptcha solver");
    }
}
