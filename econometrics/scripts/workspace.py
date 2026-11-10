"""Prepare a clean analysis folder for Claude Code: only the plugins the analysis needs are on there.

Why: with many plugins installed, Claude Code's skill listing goes over its budget and drops the
descriptions, other skills compete for the same requests, and every plugin's hooks run in every
session. A project-level .claude/settings.json can turn plugins off for one folder without touching
the rest of the computer (project settings take precedence over user settings).

Usage (Claude Code CLI on the PATH):
    python workspace.py <folder>                       # keep only econometrics on in <folder>
    python workspace.py <folder> --keep ecc@ecc        # also keep another plugin
    python workspace.py <folder> --dry-run             # show the settings without writing them

The script reads `claude plugin list --json`, writes enabledPlugins = false for every enabled plugin
that is not kept and true for the kept ones, and preserves any other keys already in the file. To undo,
delete the enabledPlugins entries (or the file). Check the result with `claude plugin list` run inside
the folder. Plugins synced from claude.ai are included; if one of them still loads, turn it off in
claude.ai or with `claude plugin disable <id> --scope project` run inside the folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import _console  # noqa: F401  (Windows-safe console output, see _console.py)

KEEP = ("econometrics",)          # a bare name keeps the plugin from any marketplace
PREPARED_KEY = "_preparedBy"      # read by setup_env.folder_settings; Claude Code ignores it


def workspace_settings(installed: list[dict], keep=KEEP, existing: dict | None = None) -> tuple[dict, list[str]]:
    """Merge enabledPlugins into existing settings: kept plugins true, every other enabled plugin false.
    installed: the list printed by `claude plugin list --json` (dicts with "id" and "enabled").
    keep: plugin ids ("name@marketplace") or bare names ("econometrics" keeps it from any marketplace).
    Returns (settings, ids turned off). Keys other than enabledPlugins are left unchanged."""
    settings = dict(existing or {})
    enabled = dict(settings.get("enabledPlugins") or {})
    off = []

    def kept(pid):
        return any(pid == k or ("@" not in k and pid.split("@")[0] == k) for k in keep)

    ids = [p.get("id") for p in installed if p.get("id")]
    for p in installed:
        pid = p.get("id")
        if not pid:
            continue
        if kept(pid):
            enabled[pid] = True
        elif p.get("enabled", True) and enabled.get(pid, True) is not False:
            enabled[pid] = False
            off.append(pid)
    for k in keep:
        if "@" in k and k not in ids:
            enabled.setdefault(k, True)
    settings["enabledPlugins"] = dict(sorted(enabled.items()))
    # A marker so the doctor knows this folder was prepared HERE, whatever the skill was installed as. Inferring
    # it from an "econometrics@…: true" key failed both ways: that key is never written when the skill is a copy
    # from the Claude app, and it is also written by enabling this skill for one project (v2.5.2 review, chat 3).
    # Claude Code ignores keys it does not know (checked against CLI 2.1.321: the settings still applied).
    settings[PREPARED_KEY] = "econometrics/workspace.py"
    return settings, off


def installed_plugins(folder: Path | None = None) -> list[dict]:
    """`claude plugin list --json`, run inside the folder when it exists (so its own settings are seen)."""
    exe = shutil.which("claude")
    if exe is None:
        raise SystemExit("Claude Code CLI ('claude') not found on the PATH: this helper is for Claude Code only.")
    cwd = folder if folder is not None and folder.is_dir() else None
    out = subprocess.run([exe, "plugin", "list", "--json"], capture_output=True, text=True, encoding="utf-8",
                         check=True, cwd=cwd).stdout
    return json.loads(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--keep", nargs="*", default=[], help="other plugin ids to keep on (name@marketplace)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    path = a.folder / ".claude" / "settings.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    settings, off = workspace_settings(installed_plugins(a.folder), keep=tuple(KEEP) + tuple(a.keep),
                                       existing=existing)
    text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    if a.dry_run:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
    print(f"Turned off in this folder: {', '.join(off) if off else 'none'}")
    print("Kept on:", ", ".join(k for k, v in settings["enabledPlugins"].items() if v))
    print("Start Claude Code inside the folder; `claude plugin list` there shows which plugins are on.")


if __name__ == "__main__":
    sys.exit(main())
