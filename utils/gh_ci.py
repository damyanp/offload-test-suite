"""Shared helpers for talking to the GitHub Actions API via the `gh` CLI.

These utilities are used by both ``runner_status.py`` (a live snapshot of what
is occupying the self-hosted runners) and ``ci_timeline.py`` (a historical
timeline of PR / scheduled CI activity and machine contention).

Requires the GitHub CLI (`gh`) to be installed and authenticated
(`gh auth login`). All API calls are issued through `gh api`.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

OWNER = "llvm"
REPO = "offload-test-suite"
VALID_VENDORS = ("intel", "amd", "nvidia", "qc")

# Workflow names that are exclusive to a specific vendor.
VENDOR_WORKFLOW_KEYWORDS = {
    "intel": "intel",
    "amd": "amd",
    "nvidia": "nvidia",
    "qc": "qc",
}


def runner_label(vendor):
    return f"hlsl-windows-{vendor}"


# ---------------------------------------------------------------------------
# Raw API access
# ---------------------------------------------------------------------------


def api_get(path):
    """Issue a GitHub API GET via `gh api` and return the parsed JSON.

    Raises subprocess.CalledProcessError on non-zero exit (e.g. 403).
    """
    result = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def api_get_paginated(path, list_key, max_items=None):
    """Fetch a paginated list endpoint and return the concatenated list.

    ``list_key`` is the field in each page's response object that holds the
    list (e.g. ``"workflow_runs"`` or ``"jobs"``). Pages are requested with
    ``per_page=100`` until a short page is returned, ``max_items`` is reached,
    or ``stop`` fires (see ``fetch_runs_window``).
    """
    sep = "&" if "?" in path else "?"
    items = []
    page = 1
    while True:
        page_path = f"{path}{sep}per_page=100&page={page}"
        data = api_get(page_path)
        page_items = data.get(list_key, [])
        items.extend(page_items)
        if len(page_items) < 100:
            break
        if max_items is not None and len(items) >= max_items:
            break
        page += 1
    if max_items is not None:
        items = items[:max_items]
    return items


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def get_all_runners():
    """Fetch all self-hosted runners registered to the repo.

    Returns a list of runner dicts, or None on error.
    """
    try:
        path = f"/repos/{OWNER}/{REPO}/actions/runners"
        return api_get_paginated(path, "runners")
    except subprocess.CalledProcessError:
        return None


def get_runners(label):
    """Fetch self-hosted runners that carry the given label.

    Returns None on error.
    """
    runners = get_all_runners()
    if runners is None:
        return None
    return [r for r in runners if label in [l["name"] for l in r.get("labels", [])]]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def parse_iso(iso_str):
    """Parse a GitHub ISO-8601 timestamp into an aware datetime (UTC)."""
    if not iso_str:
        return None
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def get_runs_by_status(status, max_items=None):
    """Fetch runs with the given status (queued / in_progress / completed)."""
    path = f"/repos/{OWNER}/{REPO}/actions/runs?status={status}"
    return api_get_paginated(path, "workflow_runs", max_items=max_items)


def fetch_runs_window(since, max_pages=50):
    """Fetch all workflow runs created at or after ``since`` (a datetime).

    Runs are returned newest-first by the API, so we page until we cross the
    ``since`` boundary. Returns the list of run dicts within the window.
    """
    path = f"/repos/{OWNER}/{REPO}/actions/runs"
    sep = "?"
    runs = []
    page = 1
    while page <= max_pages:
        page_path = f"{path}{sep}per_page=100&page={page}"
        data = api_get(page_path)
        page_runs = data.get("workflow_runs", [])
        if not page_runs:
            break
        stop = False
        for r in page_runs:
            created = parse_iso(r["created_at"])
            if created is not None and created < since:
                stop = True
                continue
            runs.append(r)
        if stop or len(page_runs) < 100:
            break
        page += 1
    return runs


def collapse_superseded(runs):
    """Keep only the most recent run per (workflow name, head branch).

    Mirrors the supersession behaviour of `cancel-in-progress` concurrency:
    a newer run for the same workflow + branch pre-empts older ones.
    Returns (kept_runs, superseded_run_ids).
    """
    latest_by_key = {}
    for r in runs:
        key = (r["name"], r.get("head_branch"))
        current = latest_by_key.get(key)
        if current is None or r["created_at"] > current["created_at"]:
            latest_by_key[key] = r
    kept = list(latest_by_key.values())
    kept_ids = {r["id"] for r in kept}
    superseded = [r["id"] for r in runs if r["id"] not in kept_ids]
    return kept, superseded


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def get_jobs(run_id):
    path = f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}/jobs"
    return api_get_paginated(path, "jobs")


def prefetch_jobs(runs, jobs_cache, max_workers=8):
    """Fetch jobs for all runs in parallel, populating ``jobs_cache``."""
    to_fetch = [r for r in runs if r["id"] not in jobs_cache]
    if not to_fetch:
        return

    def fetch_one(run_id):
        return run_id, get_jobs(run_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, r["id"]): r["id"] for r in to_fetch}
        for future in as_completed(futures):
            run_id, jobs = future.result()
            jobs_cache[run_id] = jobs


# ---------------------------------------------------------------------------
# Job classification
# ---------------------------------------------------------------------------


def job_sku_vendor(job):
    """Return the vendor implied by the matrix SKU in the job name, or None."""
    name_lower = (job.get("name") or "").lower()
    for v in VALID_VENDORS:
        if f"windows-{v}" in name_lower:
            return v
    return None


def job_matches_vendor(job, vendor, label):
    """Decide whether a job belongs to the given vendor."""
    sku_vendor = job_sku_vendor(job)
    if sku_vendor is not None:
        return sku_vendor == vendor
    return (
        label in job.get("labels", [])
        or vendor.lower() in (job.get("runner_name") or "").lower()
    )


def is_test_job(job):
    """A split-build matrix entry's test phase, named "<entry> / test"."""
    return (job.get("name") or "").endswith(" / test")


def is_build_job(job):
    """A split-build matrix entry's build phase, named "<entry> / build"."""
    return (job.get("name") or "").endswith(" / build")


def job_matrix_entry(job):
    """Return the matrix-entry name with any "/ build" or "/ test" suffix
    stripped, used to pair build and test jobs within a run."""
    name = job.get("name") or ""
    for suffix in (" / build", " / test"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def short_job_name(job):
    """Strip the "/ build" or "/ test" phase suffix and matrix prefix."""
    name = job_matrix_entry(job)
    return name.split(",")[-1].strip().rstrip(")")


def run_could_match_vendor(run, vendor):
    """Quick heuristic: can this run possibly have jobs for the given vendor?

    Scheduled/dispatch runs whose workflow name contains another vendor's
    keyword are skipped. PR runs (Execution Testing) and ambiguous runs are
    always kept.
    """
    name_lower = run["name"].lower()
    if "execution testing" in name_lower or "hlsl test" in name_lower:
        return True
    for v, kw in VENDOR_WORKFLOW_KEYWORDS.items():
        if kw in name_lower:
            return v == vendor
    return True


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def tz_abbrev(dt):
    """Short timezone abbreviation, e.g. 'PDT' instead of the long name."""
    name = dt.strftime("%Z")
    if len(name) <= 5:
        return name
    return "".join(w[0] for w in name.split())


def format_time(iso_str):
    """Convert ISO timestamp to e.g. '12:40 PM PDT / 7:40 PM UTC'."""
    dt = parse_iso(iso_str)
    h = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    utc = f"{h}:{dt.minute:02d} {ampm} UTC"
    local = dt.astimezone()
    lh = local.hour % 12 or 12
    lampm = "AM" if local.hour < 12 else "PM"
    tz = tz_abbrev(local)
    return f"{lh}:{local.minute:02d} {lampm} {tz} / {utc}"


def utcnow():
    return datetime.now(timezone.utc)
