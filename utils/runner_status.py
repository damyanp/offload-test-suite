"""
Show runner job status across all workflows in llvm/offload-test-suite.

Includes queued/in-progress runs and recently completed runs so you
can see what is (or was) occupying the runners.

Requires the GitHub CLI (`gh`) to be installed and authenticated
(`gh auth login`). API calls are issued through `gh api`.

Usage:
    python runner_status.py [vendor]

    vendor: intel | amd | nvidia | qc   (omit to show all vendors)
"""

import sys
import os
from datetime import datetime, timezone, timedelta

from gh_ci import (
    VALID_VENDORS,
    runner_label,
    get_runners,
    get_runs_by_status,
    get_jobs,
    prefetch_jobs,
    collapse_superseded,
    job_matches_vendor,
    is_test_job,
    short_job_name,
    run_could_match_vendor,
    tz_abbrev,
    format_time,
)

COMPLETED_WINDOW_HOURS = 3

# ANSI color codes per vendor
VENDOR_COLORS = {
    "intel": "\033[34m",  # blue
    "amd": "\033[31m",  # red
    "nvidia": "\033[32m",  # green
    "qc": "\033[90m",  # gray
}
RESET = "\033[0m"


def colorize(vendor, text):
    """Wrap text in the vendor's ANSI color."""
    c = VENDOR_COLORS.get(vendor, "")
    return f"{c}{text}{RESET}" if c else text


def get_runs(vendors):
    """Fetch runs that are queued, in_progress, or recently completed.

    When a single vendor is requested, only fetches runs whose workflow
    could plausibly contain jobs for that vendor.
    """
    results = []
    for status in ("queued", "in_progress"):
        results.extend(get_runs_by_status(status))

    # Also grab recently completed runs (within COMPLETED_WINDOW_HOURS)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=COMPLETED_WINDOW_HOURS)
    for r in get_runs_by_status("completed", max_items=50):
        updated = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
        if updated >= cutoff:
            results.append(r)

    # Deduplicate by run ID
    seen = set()
    unique = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    # Collapse superseded runs, runs that have been pre-empted by newer commits
    unique, _superseded = collapse_superseded(unique)

    # Pre-filter: if only one vendor requested, skip runs that clearly
    # belong to a different vendor (avoids fetching their jobs).
    if len(vendors) == 1:
        vendor = vendors[0]
        unique = [r for r in unique if run_could_match_vendor(r, vendor)]

    return unique


def print_vendor_table(vendor, runs, jobs_cache, runners_cache):
    """Print the status table for a single vendor. Returns True if any rows."""
    label = runner_label(vendor)

    # Fetch and cache runners for this label
    if label not in runners_cache:
        runners_cache[label] = get_runners(label)
    runners = runners_cache[label]
    if runners is not None:
        online = len([r for r in runners if r.get("status") == "online"])
        runner_info = f", {online}/{len(runners)} online"
    else:
        runner_info = ""

    rows = []
    active_details = []

    for run in runs:
        run_id = run["id"]
        if run_id not in jobs_cache:
            jobs_cache[run_id] = get_jobs(run_id)
        jobs = jobs_cache[run_id]

        # Match by SKU in the job name (reliable for split-build jobs), with
        # a fallback to the SKU runner label / runner name for jobs that don't
        # embed the SKU in their name.
        vendor_jobs = [j for j in jobs if job_matches_vendor(j, vendor, label)]
        if not vendor_jobs:
            continue

        title = run["display_title"]
        created = format_time(run["created_at"])

        done = len([j for j in vendor_jobs if j["status"] == "completed"])
        active = [j for j in vendor_jobs if j["status"] == "in_progress"]
        queued_jobs = [j for j in vendor_jobs if j["status"] == "queued"]
        queued_tests = len([j for j in queued_jobs if is_test_job(j)])
        queued_builds = len(queued_jobs) - queued_tests

        # Skip runs where all jobs are done (nothing active or queued)
        if not active and not queued_jobs:
            continue

        if run["event"] == "schedule":
            prefix = "[Scheduled]"
        elif run["event"] == "pull_request":
            prefix = "[PR]"
        else:
            prefix = f"[{run['event']}]"
        run_label = f"{prefix} {title} ({created})"
        rows.append((run_label, done, len(active), queued_builds, queued_tests))

        for j in active:
            active_details.append(
                (title, short_job_name(j), j.get("runner_name", "?"))
            )

    header_text = f"=== {vendor.upper()} (runner: {label}{runner_info}) ==="
    print(colorize(vendor, header_text))
    print()

    if not rows:
        print(f"No runs with {vendor} jobs found.\n")
        return False

    now_local = datetime.now().astimezone()
    local_str = now_local.strftime("%#I:%M %p ") + tz_abbrev(now_local)
    utc_str = now_local.astimezone(timezone.utc).strftime("%#I:%M %p UTC")
    timestamp = f"as of {local_str} / {utc_str}"

    col1_w = max(len(r[0]) for r in rows)
    run_col_header = f"Run ({timestamp})"
    col1_w = max(col1_w, len(run_col_header))

    done_h, active_h, qb_h, qt_h = "Done", "Active", "Queued Builds", "Queued Tests"
    header = (
        f"{run_col_header:<{col1_w}}"
        f"  {done_h:>6}  {active_h:>6}  {qb_h:>{len(qb_h)}}  {qt_h:>{len(qt_h)}}"
    )
    sep = "=" * len(header)

    print(colorize(vendor, sep))
    print(header)
    print(colorize(vendor, sep))
    for run_label, done, active, qbuilds, qtests in rows:
        active_str = str(active) if active == 0 else f"*{active}*"
        print(
            f"{run_label:<{col1_w}}"
            f"  {done:>6}  {active_str:>6}"
            f"  {qbuilds:>{len(qb_h)}}  {qtests:>{len(qt_h)}}"
        )
    print(colorize(vendor, sep))

    total_done = sum(r[1] for r in rows)
    total_active = sum(r[2] for r in rows)
    total_qbuilds = sum(r[3] for r in rows)
    total_qtests = sum(r[4] for r in rows)
    print(
        f"{'TOTAL':<{col1_w}}"
        f"  {total_done:>6}  {total_active:>6}"
        f"  {total_qbuilds:>{len(qb_h)}}  {total_qtests:>{len(qt_h)}}"
    )
    print()

    if active_details:
        print(colorize(vendor, f"Currently running on {vendor}:"))
        for title, job, runner_name in active_details:
            print(f"  -> {job}  (runner: {runner_name}, run: {title})")
    else:
        print(f"No {vendor} jobs currently running.")
    print()
    return True


def main():
    if len(sys.argv) >= 2:
        vendor = sys.argv[1].lower()
        if vendor not in VALID_VENDORS:
            print(f"Unknown vendor '{vendor}'. Choose from: {', '.join(VALID_VENDORS)}")
            print(f"Usage: python {os.path.basename(__file__)} [vendor]")
            sys.exit(1)
        vendors = [vendor]
    else:
        vendors = list(VALID_VENDORS)

    runs = get_runs(vendors)
    if not runs:
        print("No queued, in-progress, or recently completed runs found.")
        return

    runs.sort(key=lambda r: r["created_at"])

    # Fetch all jobs in parallel upfront
    jobs_cache = {}
    prefetch_jobs(runs, jobs_cache)

    runners_cache = {}

    for v in vendors:
        print_vendor_table(v, runs, jobs_cache, runners_cache)


if __name__ == "__main__":
    main()
