"""
Generate an explorable HTML timeline of PR and scheduled CI activity for
llvm/offload-test-suite, focused on machine contention across the physical
self-hosted build/test agents.

The timeline has:
  * one row per physical agent (self-hosted runner), shown even when idle,
    grouped into vendor / build-pool / macOS sections;
  * an events lane at the top with a marker per triggered run (PR push /
    schedule fire);
  * job bars (queued -> running -> done) on the owning agent's row, with the
    queue-wait portion styled distinctly;
  * build -> test dependency arrows within a run;
  * markers for cancelled jobs and superseded (cancel-in-progress) runs;
  * filtering (PR vs scheduled, vendor, workflow) and whole-run
    highlight-on-hover.

Rendering uses vis-timeline (loaded from a CDN) plus the timeline-arrows
add-on for the dependency arrows. Data is embedded in the page as JSON.

Requires the GitHub CLI (`gh`) installed and authenticated (`gh auth login`).

Usage:
    python ci_timeline.py [--days N] [--output FILE] [--refresh] [--open]
"""

import argparse
import json
import os
import sys
import tempfile
import webbrowser
from datetime import timedelta

import gh_ci


CACHE_PATH = os.path.join(tempfile.gettempdir(), "offload_ci_timeline_cache.json")

# Section ordering for the agent rows / lanes. Each entry is (id, label).
CATEGORIES = [
    ("events", "Events"),
    ("gpu-intel", "Intel GPU"),
    ("gpu-nvidia", "NVIDIA GPU"),
    ("gpu-amd", "AMD GPU"),
    ("gpu-qc", "QC GPU"),
    ("build-x64", "Build pool (Windows x64)"),
    ("build-arm64", "Build pool (ARM64)"),
    ("macos", "macOS"),
    ("hosted", "GitHub-hosted (ephemeral)"),
    ("other", "Other"),
    ("queued", "Queued (unassigned)"),
]
CATEGORY_ORDER = {cid: i for i, (cid, _label) in enumerate(CATEGORIES)}
CATEGORY_LABEL = dict(CATEGORIES)

# Categories that are single lanes rather than a section of agent rows.
LEAF_CATEGORIES = {"events", "hosted", "queued"}

# Within an agent row, jobs are split into up to three nested sub-rows, one per
# phase. PHASE_ORDER controls their top-to-bottom order (build on top).
PHASE_ORDER = ["build", "test", "other"]
PHASE_LABELS = {"build": "build", "test": "test", "other": "other"}

VENDORS = ("intel", "nvidia", "amd", "qc")


# ---------------------------------------------------------------------------
# Data gathering (with cache)
# ---------------------------------------------------------------------------


def load_cache():
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"runs": {}, "jobs": {}, "runners": []}


def save_cache(cache):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"warning: could not write cache {CACHE_PATH}: {e}", file=sys.stderr)


def gather(days, use_cache):
    """Fetch runs + jobs + runners for the last ``days`` days.

    Reuses cached jobs for runs that were already completed when cached; new
    or still-active runs are (re)fetched. Returns (runs, jobs_by_run, runners).
    """
    since = gh_ci.utcnow() - timedelta(days=days)
    cache = load_cache() if use_cache else {"runs": {}, "jobs": {}, "runners": []}

    print(f"Fetching runs since {since.isoformat()} ...", file=sys.stderr)
    runs = gh_ci.fetch_runs_window(since)
    print(f"  {len(runs)} runs in window", file=sys.stderr)

    cached_runs = cache.get("runs", {})
    cached_jobs = cache.get("jobs", {})

    jobs_by_run = {}
    need_fetch = []
    for r in runs:
        rid = str(r["id"])
        prev_run = cached_runs.get(rid)
        prev_jobs = cached_jobs.get(rid)
        if (
            use_cache
            and prev_jobs is not None
            and prev_run is not None
            and prev_run.get("status") == "completed"
        ):
            jobs_by_run[r["id"]] = prev_jobs
        else:
            need_fetch.append(r)

    print(
        f"  reusing jobs for {len(runs) - len(need_fetch)} runs, "
        f"fetching jobs for {len(need_fetch)} ...",
        file=sys.stderr,
    )
    fetched = {}
    gh_ci.prefetch_jobs(need_fetch, fetched)
    jobs_by_run.update(fetched)

    runners = gh_ci.get_all_runners()
    if runners is None:
        runners = cache.get("runners", []) if use_cache else []
        print("  warning: could not fetch runners; using cached/empty", file=sys.stderr)

    # Persist an updated cache (only for the current window).
    new_cache = {
        "runs": {str(r["id"]): r for r in runs},
        "jobs": {str(rid): jobs_by_run[rid] for rid in jobs_by_run},
        "runners": runners,
        "fetched_at": gh_ci.utcnow().isoformat(),
    }
    save_cache(new_cache)

    return runs, jobs_by_run, runners


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def categorize_labels(labels):
    """Map a runner's labels to a section id and (optional) vendor.

    Registered-runner labels are dicts ({"name": ...}); job labels are plain
    strings. Handle both.
    """
    ls = []
    for l in labels or []:
        if isinstance(l, dict):
            ls.append(str(l.get("name", "")).lower())
        else:
            ls.append(str(l).lower())
    for v in VENDORS:
        if f"hlsl-windows-{v}" in ls:
            return f"gpu-{v}", v
    if "hlsl-macos" in ls or "macos" in ls:
        return "macos", None
    if "arm64" in ls:
        return "build-arm64", None
    if "x64" in ls or "windows" in ls:
        return "build-x64", None
    return "other", None


def categorize_runner_name(name):
    """Best-effort section/vendor from a runner name when labels are unknown."""
    n = (name or "").lower()
    if n.startswith("github actions "):
        return "hosted", None
    for v in VENDORS:
        if v in n:
            return f"gpu-{v}", v
    return "other", None


def event_kind(run):
    ev = run.get("event")
    return {
        "pull_request": "pr",
        "pull_request_target": "pr",
        "schedule": "schedule",
        "workflow_dispatch": "dispatch",
        "push": "push",
    }.get(ev, ev or "other")


EVENT_PREFIX = {
    "pr": "[PR]",
    "schedule": "[Scheduled]",
    "dispatch": "[Dispatch]",
    "push": "[Push]",
}


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_duration(start, end):
    a = gh_ci.parse_iso(start)
    b = gh_ci.parse_iso(end)
    if a is None or b is None:
        return "?"
    secs = max(0, int((b - a).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_model(runs, jobs_by_run, runners, days):
    now_iso = gh_ci.utcnow().isoformat()

    # --- Agent rows -------------------------------------------------------
    # name -> (category, vendor). Seed from registered runners so idle agents
    # still get a row.
    agents = {}
    for r in runners or []:
        cat, vendor = categorize_labels(r.get("labels"))
        agents[r["name"]] = (cat, vendor)

    runs_by_id = {r["id"]: r for r in runs}
    _kept, superseded_ids = gh_ci.collapse_superseded(runs)
    superseded_ids = set(superseded_ids)

    items = []
    arrows = []
    item_ids = set()
    used_categories = set(["events"])
    workflows = set()
    # name -> set of phases ("build"/"test"/"other") actually seen on that agent,
    # used to build up to three nested sub-rows per physical agent.
    agent_phases = {}

    def add_agent(name):
        """Register a runner and return (kind, key).

        kind is "hosted" for ephemeral GitHub-hosted runners (single lane) or
        "agent" for a physical self-hosted agent, in which case key is its name.
        """
        if name.startswith("GitHub Actions "):
            used_categories.add("hosted")
            return ("hosted", None)
        if name not in agents:
            agents[name] = categorize_runner_name(name)
        used_categories.add(agents[name][0])
        return ("agent", name)

    def agent_vendor(name):
        if name in agents:
            return agents[name][1]
        return None

    for run in runs:
        rid = run["id"]
        ekind = event_kind(run)
        wf = run.get("name") or ""
        workflows.add(wf)
        title = run.get("display_title") or wf
        run_url = run.get("html_url")
        superseded = rid in superseded_ids
        run_cancelled = run.get("conclusion") == "cancelled"
        prefix = EVENT_PREFIX.get(ekind, f"[{ekind}]")

        # Events-lane marker for the triggered run.
        ev_tip = (
            f"<b>{esc(prefix)} {esc(title)}</b><br>"
            f"workflow: {esc(wf)}<br>"
            f"event: {esc(run.get('event'))}<br>"
            f"branch: {esc(run.get('head_branch'))}<br>"
            f"triggered: {esc(gh_ci.format_time(run['created_at']))}"
        )
        if superseded:
            ev_tip += "<br><b>superseded (cancel-in-progress)</b>"
        items.append(
            {
                "id": f"run-{rid}",
                "group": "events",
                "start": run["created_at"],
                "type": "point",
                "content": esc(f"{prefix} {title[:48]}"),
                "className": "ev ev-" + ekind + (" run-cancelled" if run_cancelled else ""),
                "title": ev_tip,
                "_event": ekind,
                "_vendor": "",
                "_workflow": wf,
                "_runId": rid,
                "_url": run_url,
                "_cancelled": False,
                "_superseded": superseded,
            }
        )

        jobs = jobs_by_run.get(rid, [])

        # Track per matrix-entry build/test item ids for dependency arrows.
        entry_build = {}
        entry_test = {}

        for job in jobs:
            jid = job["id"]
            name = job.get("name") or ""
            status = job.get("status")
            conclusion = job.get("conclusion")
            runner_name = job.get("runner_name")
            created = job.get("created_at")
            started = job.get("started_at")
            completed = job.get("completed_at")
            job_url = job.get("html_url")
            vendor = gh_ci.job_sku_vendor(job)

            # Skipped jobs (excluded matrix combos / false `if:`) never occupy
            # a machine -- don't clutter the timeline with them.
            if conclusion == "skipped":
                continue

            phase = (
                "test"
                if gh_ci.is_test_job(job)
                else "build" if gh_ci.is_build_job(job) else "other"
            )
            concl = conclusion or status or "unknown"
            short = gh_ci.short_job_name(job)

            has_runner = bool(runner_name and str(runner_name).strip())
            is_cancelled = conclusion == "cancelled"

            if started and has_runner:
                kind, aname = add_agent(runner_name)
                if kind == "hosted":
                    group = "hosted"
                else:
                    group = f"agent::{aname}::{phase}"
                    agent_phases.setdefault(aname, set()).add(phase)
                if vendor is None:
                    vendor = agent_vendor(runner_name)
                end = completed or now_iso
                tip = (
                    f"<b>{esc(name)}</b><br>"
                    f"run: {esc(prefix)} {esc(title)}<br>"
                    f"workflow: {esc(wf)}<br>"
                    f"agent: {esc(runner_name)}<br>"
                    f"status: {esc(concl)}<br>"
                    f"queued: {esc(gh_ci.format_time(created)) if created else '?'}<br>"
                    f"started: {esc(gh_ci.format_time(started))}<br>"
                    f"finished: {esc(gh_ci.format_time(completed)) if completed else '(running)'}<br>"
                    f"queue-wait: {esc(fmt_duration(created, started)) if created else '?'}<br>"
                    f"run-time: {esc(fmt_duration(started, end))}"
                )
                job_item_id = f"job-{jid}"
                items.append(
                    {
                        "id": job_item_id,
                        "group": group,
                        "start": started,
                        "end": end,
                        "content": esc(short),
                        "className": f"job ph-{phase} concl-{esc(concl)}"
                        + (" running" if not completed else ""),
                        "title": tip,
                        "_event": ekind,
                        "_vendor": vendor or "",
                        "_workflow": wf,
                        "_runId": rid,
                        "_url": job_url,
                        "_cancelled": is_cancelled,
                        "_superseded": superseded,
                        "_queued": False,
                    }
                )
                item_ids.add(job_item_id)

                # Consolidated activity box on the agent's heading (parent) row,
                # colour-coded by trigger: blue = PR, green = Scheduled, grey =
                # other. Idle agents get no box (bare background).
                if kind == "agent":
                    busy_kind = (
                        "pr" if ekind == "pr"
                        else "schedule" if ekind == "schedule"
                        else "other"
                    )
                    items.append(
                        {
                            "id": f"busy-{jid}",
                            "group": f"agent::{aname}",
                            "start": started,
                            "end": end,
                            "content": "",
                            "className": f"busy busy-{busy_kind}"
                            + (" running" if not completed else ""),
                            "title": tip,
                            "_event": ekind,
                            "_vendor": vendor or "",
                            "_workflow": wf,
                            "_runId": rid,
                            "_url": job_url,
                            "_cancelled": is_cancelled,
                            "_superseded": superseded,
                            "_queued": False,
                        }
                    )
                    item_ids.add(f"busy-{jid}")

                # Queue-wait segment (distinct styling) on the same row.
                if created and gh_ci.parse_iso(started) and gh_ci.parse_iso(created):
                    wait_s = (
                        gh_ci.parse_iso(started) - gh_ci.parse_iso(created)
                    ).total_seconds()
                    if wait_s >= 2:
                        items.append(
                            {
                                "id": f"qw-{jid}",
                                "group": group,
                                "start": created,
                                "end": started,
                                "content": "",
                                "className": "qwait",
                                "title": f"queue-wait: {esc(fmt_duration(created, started))}<br>{esc(short)}",
                                "_event": ekind,
                                "_vendor": vendor or "",
                                "_workflow": wf,
                                "_runId": rid,
                                "_url": job_url,
                                "_cancelled": is_cancelled,
                                "_superseded": superseded,
                                "_queued": True,
                            }
                        )

                # Cancellation marker at the end of a cancelled bar.
                if conclusion == "cancelled" and completed:
                    items.append(
                        {
                            "id": f"cx-{jid}",
                            "group": group,
                            "start": completed,
                            "type": "point",
                            "content": "\u2715",
                            "className": "cancel-x",
                            "title": f"cancelled: {esc(name)}",
                            "_event": ekind,
                            "_vendor": vendor or "",
                            "_workflow": wf,
                            "_runId": rid,
                            "_url": job_url,
                            "_cancelled": True,
                            "_superseded": superseded,
                            "_queued": False,
                        }
                    )
            else:
                # Job never occupied a real agent: still queued/waiting, or
                # cancelled before it was assigned to a runner. Show it in the
                # unassigned lane spanning queued-time -> end (or now).
                used_categories.add("queued")
                job_item_id = f"job-{jid}"
                end = completed or now_iso
                if conclusion:
                    end_desc = f"ended ({esc(concl)}): {esc(gh_ci.format_time(completed)) if completed else '?'}"
                else:
                    end_desc = "not yet started"
                tip = (
                    f"<b>{esc(name)}</b><br>"
                    f"run: {esc(prefix)} {esc(title)}<br>"
                    f"workflow: {esc(wf)}<br>"
                    f"status: {esc(concl)} (never assigned an agent)<br>"
                    f"queued: {esc(gh_ci.format_time(created)) if created else '?'}<br>"
                    f"{end_desc}<br>"
                    f"waiting: {esc(fmt_duration(created, end)) if created else '?'}"
                )
                items.append(
                    {
                        "id": job_item_id,
                        "group": "queued",
                        "start": created or now_iso,
                        "end": end,
                        "content": esc(short),
                        "className": f"job ph-{phase} queued-job"
                        + (" concl-cancelled" if conclusion == "cancelled" else ""),
                        "title": tip,
                        "_event": ekind,
                        "_vendor": vendor or "",
                        "_workflow": wf,
                        "_runId": rid,
                        "_url": job_url,
                        "_cancelled": is_cancelled,
                        "_superseded": superseded,
                        "_queued": True,
                    }
                )
                item_ids.add(job_item_id)

            entry = gh_ci.job_matrix_entry(job)
            if phase == "build":
                entry_build[entry] = f"job-{jid}"
            elif phase == "test":
                entry_test[entry] = f"job-{jid}"

        # build -> test arrows within this run.
        for entry, build_id in entry_build.items():
            test_id = entry_test.get(entry)
            if test_id and build_id in item_ids and test_id in item_ids:
                arrows.append(
                    {
                        "id": f"dep-{build_id}-{test_id}",
                        "id_item_1": build_id,
                        "id_item_2": test_id,
                    }
                )

    groups = build_groups(agents, agent_phases, used_categories)

    return {
        "groups": groups,
        "items": items,
        "arrows": arrows,
        "workflows": sorted(w for w in workflows if w),
        "windowStart": (gh_ci.utcnow() - timedelta(days=days)).isoformat(),
        "now": now_iso,
    }


def build_groups(agents, agent_phases, used_categories):
    """Build the vis-timeline groups (nested sections + agent rows).

    Each physical agent becomes a parent group with up to three nested child
    rows (build / test / other), one per phase actually seen on that agent, so
    build and test never share a line. Idle agents (no jobs) render as a single
    plain row.
    """
    groups = []
    order = 0

    # Group agent names by category.
    by_cat = {}
    for name, (cat, _vendor) in agents.items():
        by_cat.setdefault(cat, []).append(name)

    for cid, label in CATEGORIES:
        if cid in LEAF_CATEGORIES:
            if cid in used_categories:
                groups.append(
                    {
                        "id": cid,
                        "content": label,
                        "order": order,
                        "className": f"cat cat-{cid}",
                    }
                )
                order += 1
            continue

        names = sorted(by_cat.get(cid, []))
        if not names:
            continue
        parent_order = order
        order += 1
        cat_children = [f"agent::{name}" for name in names]
        groups.append(
            {
                "id": f"cat::{cid}",
                "content": label,
                "order": parent_order,
                "className": f"cat cat-{cid}",
                "nestedGroups": cat_children,
            }
        )
        for name in names:
            phases = agent_phases.get(name, set())
            phase_children = [ph for ph in PHASE_ORDER if ph in phases]
            agent_group = {
                "id": f"agent::{name}",
                "content": esc(name),
                "order": order,
                "className": f"agent agent-{cid}",
            }
            order += 1
            if phase_children:
                agent_group["nestedGroups"] = [
                    f"agent::{name}::{ph}" for ph in phase_children
                ]
                groups.append(agent_group)
                for ph in phase_children:
                    groups.append(
                        {
                            "id": f"agent::{name}::{ph}",
                            "content": PHASE_LABELS[ph],
                            "order": order,
                            "className": f"phase-row phase-{ph}",
                        }
                    )
                    order += 1
            else:
                # Idle agent: single plain row, no nested phases.
                groups.append(agent_group)

    return groups


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_html(model):
    data_json = json.dumps(model)
    return HTML_TEMPLATE.replace("__DATA__", data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Offload CI Contention Timeline</title>
<link href="https://cdn.jsdelivr.net/npm/vis-timeline@7.7.3/styles/vis-timeline-graph2d.min.css" rel="stylesheet">
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; }
  header { padding: 8px 12px; border-bottom: 1px solid #ccc; }
  header h1 { font-size: 16px; margin: 0 0 6px; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 12px; }
  .controls fieldset { border: 1px solid #bbb; border-radius: 5px; padding: 2px 8px; margin: 0; }
  .controls legend { padding: 0 4px; color: #666; }
  .controls label { margin-right: 8px; white-space: nowrap; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: #555; align-items: center; }
  .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; margin-right: 3px; }
  #timeline { }

  /* Job bars: fill by phase (build / test / other); conclusion shown via
     border + hatch overlay so the phase colour is always distinguishable. */
  .vis-item.job { border-color: #5a9a5a; background-color: #d7ecdd; color: #123; }
  .vis-item.ph-build { background-color: #f1e4bf; border-color: #b8912e; }
  .vis-item.ph-test  { background-color: #cfe0f9; border-color: #3f6fb0; }
  .vis-item.ph-other { background-color: #e3e7ea; border-color: #7a8790; }
  .vis-item.concl-failure { border-color: #c0392b; border-width: 2px;
      background-image: repeating-linear-gradient(45deg,rgba(192,57,43,.22),rgba(192,57,43,.22) 4px,transparent 4px,transparent 8px); }
  .vis-item.concl-cancelled { opacity: .6; border-color: #999; color: #888;
      background-image: repeating-linear-gradient(45deg,transparent,transparent 4px,rgba(120,120,120,.35) 4px,rgba(120,120,120,.35) 8px); }
  .vis-item.running { border-style: dashed; }
  .vis-item.queued-job { background-color: #fff3cd; border-color: #cba300; color: #664d00; }

  /* Queue-wait segment */
  .vis-item.qwait { background-color: transparent; border-color: #cba300;
      background-image: repeating-linear-gradient(45deg,rgba(203,163,0,.15),rgba(203,163,0,.15) 3px,transparent 3px,transparent 6px);
      border-style: dotted; }

  /* Event markers */
  .vis-item.ev { border-width: 2px; }
  .vis-item.ev-pr .vis-item-content::before { content: ""; }
  .vis-item.ev-pr { color: #06c; }
  .vis-item.ev-schedule { color: #582; }
  .vis-item.ev-dispatch { color: #94a; }
  .vis-item.run-cancelled { text-decoration: line-through; opacity: .7; }

  .vis-item.cancel-x { color: #b00; font-weight: bold; border: none; background: transparent; }

  /* Consolidated activity boxes on the agent heading row */
  .vis-item.busy { border-radius: 2px; height: 12px; border-width: 1px; }
  .vis-item.busy-pr { background-color: #3f6fb0; border-color: #2b5286; }
  .vis-item.busy-schedule { background-color: #4a9a4a; border-color: #337033; }
  .vis-item.busy-other { background-color: #9aa4ad; border-color: #74808a; }
  .vis-item.busy.running { border-style: dashed; }

  /* Section (category) group headers */
  .vis-label.cat, .vis-labelset .vis-label.cat { font-weight: 600; background: rgba(0,0,0,.05); }
  .vis-label.agent { font-size: 11px; }
  /* Nested phase sub-rows (build / test / other) */
  .vis-label.phase-row { font-size: 10px; color: #777; }
  .vis-label.phase-row .vis-inner { padding-left: 6px; }
  .vis-label.phase-build .vis-inner { border-left: 3px solid #b8912e; }
  .vis-label.phase-test  .vis-inner { border-left: 3px solid #3f6fb0; }
  .vis-label.phase-other .vis-inner { border-left: 3px solid #7a8790; }

  /* Highlight-on-hover: dim everything, keep hovered run bright */
  #timeline.dimming .vis-item { opacity: .2; }
  #timeline.dimming .vis-item.hl { opacity: 1; }

  .hint { font-size: 11px; color: #888; margin-left: auto; }
</style>
</head>
<body>
<header>
  <h1>Offload CI Contention Timeline
      <span class="hint">hover to preview &middot; click to pin &middot; double-click to open</span>
      <span class="hint" id="meta"></span></h1>
  <div class="controls">
    <fieldset><legend>Event</legend><span id="event-filters"></span></fieldset>
    <fieldset><legend>Vendor</legend><span id="vendor-filters"></span></fieldset>
    <fieldset><legend>Workflow</legend>
      <select id="workflow-filter"><option value="">(all)</option></select>
    </fieldset>
    <fieldset><legend>Agent</legend>
      <input id="agent-filter" type="text" placeholder="name contains..." size="14">
    </fieldset>
    <fieldset><legend>Cancelled</legend>
      <label><input id="cancelled-filter" type="checkbox" checked> Show cancelled</label>
    </fieldset>
    <fieldset><legend>Superseded</legend>
      <label><input id="superseded-filter" type="checkbox"> Show superseded</label>
    </fieldset>
    <fieldset><legend>Queued</legend>
      <label><input id="queued-filter" type="checkbox"> Show queued</label>
    </fieldset>
    <div class="legend">
      <span><span class="swatch" style="background:#f1e4bf"></span>build</span>
      <span><span class="swatch" style="background:#cfe0f9"></span>test</span>
      <span><span class="swatch" style="background:#e3e7ea"></span>other</span>
      <span><span class="swatch" style="background:repeating-linear-gradient(45deg,#e7b4ad,#e7b4ad 3px,transparent 3px,transparent 6px);border:1px solid #c0392b"></span>failed</span>
      <span><span class="swatch" style="background:#fff3cd"></span>queued</span>
      <span><span class="swatch" style="background:repeating-linear-gradient(45deg,#cba300,#cba300 3px,transparent 3px,transparent 6px)"></span>queue-wait</span>
      <span style="color:#9c0000">&rarr; build&rarr;test</span>
    </div>
  </div>
</header>
<div id="timeline"></div>

<script src="https://cdn.jsdelivr.net/npm/vis-timeline@7.7.3/standalone/umd/vis-timeline-graph2d.min.js"></script>
<script type="module">
import Arrow from 'https://cdn.jsdelivr.net/npm/timeline-arrows@4.8.0/+esm';

const DATA = __DATA__;
const vis = window.vis;

// Master copy of all items for filtering.
const MASTER = DATA.items;
const runItems = {};        // runId -> [itemId]
const itemRun = {};         // itemId -> runId
for (const it of MASTER) {
  (runItems[it._runId] = runItems[it._runId] || []).push(it.id);
  itemRun[it.id] = it._runId;
}

// Within an agent row, jobs are split into up to three nested sub-rows: build
// (top), test, then other. These are real nested child groups built server-side
// (ids "agent::NAME::PHASE"), ordered via the `order` field.
const groups = new vis.DataSet(DATA.groups);
const items = new vis.DataSet(MASTER);

const container = document.getElementById('timeline');
const now = new Date(DATA.now);
const dayAgo = new Date(now.getTime() - 24*3600*1000);
const winStart = new Date(DATA.windowStart);

const options = {
  stack: true,
  stackSubgroups: true,
  groupOrder: 'order',
  orientation: 'top',
  zoomMin: 1000*30,
  zoomMax: 1000*3600*24*14,
  min: winStart,
  max: new Date(now.getTime() + 3600*1000),
  start: dayAgo < winStart ? winStart : dayAgo,
  end: new Date(now.getTime() + 5*60*1000),
  margin: { item: { horizontal: 0, vertical: 2 }, axis: 4 },
  tooltip: { followMouse: true, overflowMethod: 'flip' },
  template: null,
  height: 'calc(100vh - 78px)',
};

const timeline = new vis.Timeline(container, items, groups, options);

// Dependency arrows (build -> test). Created once with the full set; the
// library hides an arrow automatically when either endpoint item is filtered
// out of the DataSet, so we never need to recreate it.
const arrow = new Arrow(timeline, DATA.arrows, {
  color: '#9c0000', strokeWidth: 2, hideWhenItemsNotVisible: true,
});

// Highlighting: hover previews a run; single click freezes the selection;
// double click opens the run/job on GitHub. Clicking empty space clears.
let frozenRun = null;   // runId whose highlight is pinned, or null
let hlIds = [];         // item ids currently carrying the 'hl' class

function clearHl() {
  if (hlIds.length) {
    const upd = hlIds.map(id => {
      const it = items.get(id);
      if (!it) return null;
      return { id, className: it.className.replace(/ hl\b/, '') };
    }).filter(Boolean);
    items.update(upd);
    hlIds = [];
  }
  container.classList.remove('dimming');
}

function showHl(runId) {
  clearHl();
  if (runId == null) return;
  hlIds = (runItems[runId] || []).filter(id => items.get(id));
  if (!hlIds.length) return;
  container.classList.add('dimming');
  items.update(hlIds.map(id => {
    const it = items.get(id);
    return { id, className: it.className + ' hl' };
  }));
}

timeline.on('itemover', props => {
  const runId = itemRun[props.item];
  if (runId == null) return;
  showHl(runId);
});
timeline.on('itemout', () => {
  // Restore the pinned selection (or clear if nothing is pinned).
  showHl(frozenRun);
});

// Single click: pin (freeze) the clicked item's run, or clear on background.
// Double click: follow the link to the run/job on GitHub. We detect the
// double click from click timing (Hammer's own doubletap is unreliable for
// synthetic input) and also honour vis' native doubleClick; a shared debounced
// opener guarantees at most one tab per gesture.
let lastOpen = 0;
function openItem(id) {
  if (id == null) return;
  const it = items.get(id);
  if (!it || !it._url) return;
  const now = Date.now();
  if (now - lastOpen < 500) return;
  lastOpen = now;
  window.open(it._url, '_blank');
}

let lastClickItem = null, lastClickTime = 0;
// Track pointer-down position so we can distinguish a deliberate click from a
// pan drag: panning should never clear/change the pinned selection.
let downX = 0, downY = 0, moved = false;
const MOVE_THRESHOLD = 5; // px
container.addEventListener('pointerdown', e => {
  downX = e.pageX; downY = e.pageY; moved = false;
});
container.addEventListener('pointermove', e => {
  if (Math.abs(e.pageX - downX) > MOVE_THRESHOLD ||
      Math.abs(e.pageY - downY) > MOVE_THRESHOLD) moved = true;
});
timeline.on('click', props => {
  // A drag/pan gesture also fires 'click' on release; ignore it so panning
  // keeps the current selection intact.
  if (moved) return;
  const id = props.item;
  const now = Date.now();
  const isDouble = id != null && id === lastClickItem && (now - lastClickTime) < 350;
  lastClickItem = id;
  lastClickTime = now;
  // Pin the clicked item's run, or clear the pin on a background click.
  frozenRun = id != null ? itemRun[id] : null;
  showHl(frozenRun);
  if (isDouble) openItem(id);
});

// Native double click (real-browser gesture) -> open, debounced against the above.
timeline.on('doubleClick', props => openItem(props.item));

// ---- Filters ----
const EVENTS = [['pr','PR'],['schedule','Scheduled'],['dispatch','Dispatch'],['push','Push']];
const EVENT_KEYS = new Set(EVENTS.map(e=>e[0]));
const VENDORS = [['intel','Intel'],['nvidia','NVIDIA'],['amd','AMD'],['qc','QC']];
const state = { events: new Set(EVENTS.map(e=>e[0])), vendors: null, workflow: '', agent: '', showCancelled: true, showSuperseded: false, showQueued: false };

function makeChecks(host, list, key) {
  const el = document.getElementById(host);
  for (const [val,label] of list) {
    const id = host+'-'+val;
    const wrap = document.createElement('label');
    const cb = document.createElement('input');
    cb.type='checkbox'; cb.checked=true; cb.value=val; cb.id=id;
    cb.addEventListener('change', () => {
      const set = state[key] || (state[key]=new Set());
      if (cb.checked) set.add(val); else set.delete(val);
      applyFilters();
    });
    wrap.appendChild(cb); wrap.appendChild(document.createTextNode(' '+label));
    el.appendChild(wrap);
  }
}
state.vendors = new Set(VENDORS.map(v=>v[0]));
makeChecks('event-filters', EVENTS, 'events');
makeChecks('vendor-filters', VENDORS, 'vendors');

const wfSel = document.getElementById('workflow-filter');
for (const wf of DATA.workflows) {
  const o = document.createElement('option'); o.value=wf; o.textContent=wf; wfSel.appendChild(o);
}
wfSel.addEventListener('change', () => { state.workflow = wfSel.value; applyFilters(); });
const agentInput = document.getElementById('agent-filter');
agentInput.addEventListener('input', () => { state.agent = agentInput.value.toLowerCase(); applyFilters(); });
const cancelledInput = document.getElementById('cancelled-filter');
cancelledInput.addEventListener('change', () => { state.showCancelled = cancelledInput.checked; applyFilters(); });
const supersededInput = document.getElementById('superseded-filter');
supersededInput.addEventListener('change', () => { state.showSuperseded = supersededInput.checked; applyFilters(); });
const queuedInput = document.getElementById('queued-filter');
queuedInput.addEventListener('change', () => { state.showQueued = queuedInput.checked; applyFilters(); });

function visible(it) {
  if (!state.showCancelled && it._cancelled) return false;
  if (!state.showSuperseded && it._superseded) return false;
  if (!state.showQueued && it._queued) return false;
  // Event-lane markers: filter by event only. Only apply the toggle to
  // known event kinds so any unrecognised event type is always shown.
  if (it._event && EVENT_KEYS.has(it._event) && !state.events.has(it._event)) return false;
  // Vendor filter only applies to items that have a vendor.
  if (it._vendor && !state.vendors.has(it._vendor)) return false;
  if (state.workflow && it._workflow !== state.workflow) return false;
  if (state.agent && it.group !== 'events') {
    const g = String(it.group||'');
    if (!g.toLowerCase().includes(state.agent)) return false;
  }
  return true;
}

function applyFilters() {
  const keep = MASTER.filter(visible);
  items.clear();
  items.add(keep);
  hlIds = [];
  showHl(frozenRun);
}

// Apply the initial filter state (superseded runs hidden by default).
applyFilters();

document.getElementById('meta').textContent =
  `${MASTER.length} items \u00b7 ${DATA.groups.filter(g=>String(g.id).split('::').length===2 && String(g.id).startsWith('agent::')).length} agents \u00b7 window from ${winStart.toISOString().slice(0,10)}`;

// Expose objects for debugging / power users (e.g. window.ciTimeline.timeline.fit()).
window.ciTimeline = { timeline, items, groups, master: MASTER, state, getFrozen: () => frozenRun };
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days of history to cover (default: 7)"
    )
    parser.add_argument(
        "--output",
        default="ci_timeline.html",
        help="Output HTML file (default: ci_timeline.html)",
    )
    parser.add_argument(
        "--refresh",
        "--no-cache",
        dest="refresh",
        action="store_true",
        help="Ignore the local cache and fetch everything fresh",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the generated file in a browser"
    )
    args = parser.parse_args(argv[1:])

    runs, jobs_by_run, runners = gather(args.days, use_cache=not args.refresh)
    if not runs:
        print("No runs found in the requested window.", file=sys.stderr)
        return 1

    model = build_model(runs, jobs_by_run, runners, args.days)
    html = render_html(model)

    out = os.path.abspath(args.output)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote timeline: {out}")
    print(
        f"  {len(model['items'])} items, {len(model['arrows'])} dependency arrows, "
        f"{len(model['groups'])} groups"
    )
    if args.open:
        webbrowser.open(f"file://{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
