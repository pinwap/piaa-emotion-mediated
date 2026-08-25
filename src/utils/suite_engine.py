"""Core Reusable Suite Engine for Experiment Execution, Tracking, and Archiving.

Provides the foundational infrastructure to define, run, track, and package any experiment suite.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"


@dataclass
class SuiteStep:
    id: int
    codename: str
    title: str
    desc: str
    cmd: list[str]
    folder: str = ""

    def __post_init__(self):
        if not self.folder:
            self.folder = f"{self.id}_{self.codename}"


@dataclass
class Suite:
    name: str
    title: str
    desc: str
    steps: list[SuiteStep] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return OUTPUT_DIR / self.name

    @property
    def zip_path(self) -> Path:
        return self.output_dir / f"{self.name}_all_runs.zip"


def snapshot_output(suite_name: str) -> dict[Path, float]:
    """Capture mtimes of all files in output/ (excluding the current suite folder)."""
    snap = {}
    if not OUTPUT_DIR.exists():
        return snap
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.name != ".DS_Store" and suite_name not in p.parts:
            try:
                snap[p] = p.stat().st_mtime
            except OSError:
                pass
    return snap


def check_step_status(suite: Suite, step: SuiteStep) -> tuple[bool, list[str]]:
    """Check if a step has completed results inside output/<suite_name>/<folder>/ or output/<suite_name>/modal/<folder>/."""
    candidates = [
        suite.output_dir / step.folder,
        suite.output_dir / "modal" / step.folder,
        suite.output_dir / step.codename,
        suite.output_dir / "modal" / step.codename,
    ]
    for target_dir in candidates:
        if target_dir.exists():
            files = sorted([f.name for f in target_dir.glob("*.csv")])
            if files:
                rel_path = target_dir.relative_to(OUTPUT_DIR)
                return True, [f"{rel_path}/{f}" for f in files]
    return False, []


def render_suite_table(suite: Suite):
    """Render a beautiful Rich table for any suite."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    completed_count = 0
    total_steps = len(suite.steps)

    if use_rich:
        table = Table(
            title=f"[bold white on blue]  EXPERIMENT SUITE: {suite.name.upper()}  [/] ([dim]{suite.title}[/])",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
            expand=True
        )
        table.add_column("#", justify="center", style="bold white", width=4)
        table.add_column("Status", justify="center", width=14)
        table.add_column("Codename", style="bold magenta", width=16)
        table.add_column("Folder", style="cyan", width=18)
        table.add_column("Details & Description", style="white", ratio=2)
        table.add_column("Command Preview", style="dim", ratio=2)
        table.add_column("Output Files", style="green", ratio=1)

        for s in suite.steps:
            done, files = check_step_status(suite, s)
            if done:
                completed_count += 1
                status = "[bold green]✓ COMPLETED[/]"
                out_str = "\n".join([Path(f).name for f in files])
            else:
                status = "[bold yellow]⏳ NOT RUN[/]"
                out_str = "[dim]None[/]"

            cmd_str = " ".join(s.cmd if isinstance(s.cmd, list) else [s.cmd])
            details = f"[bold]{s.title}[/]\n[dim]{s.desc}[/]"

            table.add_row(
                str(s.id),
                status,
                s.codename,
                s.folder,
                details,
                cmd_str,
                out_str
            )

        pct = (completed_count / total_steps) * 100 if total_steps else 0
        console.print()
        console.print(table)

        progress_color = "green" if completed_count == total_steps and total_steps > 0 else "yellow"
        summary_panel = Panel(
            f"[{progress_color}]Progress: [bold]{completed_count}/{total_steps}[/] steps completed ([bold]{pct:.1f}%[/])[/]\n"
            f"[dim]Output Directory:[/] [cyan]output/{suite.name}/[/]  |  [dim]Consolidated Zip:[/] [cyan]{suite.zip_path.name}[/]\n\n"
            f"[bold white]Quick Commands:[/] \n"
            f"  • Run all steps in suite:     [green]uv run run_{suite.name}.py --all[/]\n"
            f"  • Run specific codename:     [green]uv run run_{suite.name}.py --run {suite.steps[0].codename if suite.steps else 'step'}[/]\n"
            f"  • Package results into zip:   [green]uv run run_{suite.name}.py --zip[/]",
            title=f"[bold]Suite '{suite.name}' Summary[/]",
            border_style=progress_color,
            box=box.ROUNDED
        )
        console.print(summary_panel)
        console.print()
    else:
        print("=" * 90)
        print(f"SUITE: {suite.name} - {suite.title}")
        print("=" * 90)
        for s in suite.steps:
            done, files = check_step_status(suite, s)
            status_str = "[DONE] ✓" if done else "[PENDING] ⏳"
            print(f"[{s.id}] {status_str:<10} | {s.codename:<14} | {s.folder:<16} | {s.title}")
            print(f"    Command: {' '.join(s.cmd)}")
            if done:
                print(f"    Outputs: {', '.join(files)}")
            print("-" * 90)
        pct = (completed_count / total_steps) * 100 if total_steps else 0
        print(f"Progress: {completed_count}/{total_steps} ({pct:.1f}%)")
        print("=" * 90)


def package_suite_zip(suite: Suite, from_modal: bool = False):
    """Package all output files for a suite into a zip archive.

    If from_modal is True:
        Packages files from output/<suite_name>/modal/ (or fallback to output/<suite_name>/).
        The internal archive structure is clean: <suite_name>/<step_folder>/...
        WITHOUT any 'modal/' subdirectory inside the zip archive!
        Archive target: output/<suite_name>/<suite_name>_modal_all_runs.zip
    If from_modal is False:
        Packages files from output/<suite_name>/ (excluding the modal/ directory).
        Archive target: output/<suite_name>/<suite_name>_all_runs.zip
    """
    if from_modal:
        modal_dir = suite.output_dir / "modal"
        suite_dir = modal_dir if modal_dir.exists() else suite.output_dir
        zip_path = suite.output_dir / f"{suite.name}_modal_all_runs.zip"
        mode_label = "MODAL CLOUD (stripped 'modal/' subfolder)"
    else:
        suite_dir = suite.output_dir
        zip_path = suite.zip_path
        mode_label = "LOCAL (excluding 'modal/' subfolder)"

    if not suite_dir.exists():
        print(f"Error: Directory '{suite_dir.relative_to(ROOT)}' does not exist yet.")
        sys.exit(1)

    raw_files = [p for p in suite_dir.rglob("*") if p.is_file() and not p.name.startswith(".") and p.suffix != ".zip"]

    files_to_zip = []
    for p in raw_files:
        rel = p.relative_to(suite_dir)
        # If packaging local outputs, ignore files inside modal/ subdirectory
        if not from_modal and "modal" in rel.parts:
            continue
        files_to_zip.append((p, rel))

    if not files_to_zip:
        print(f"Warning: No output files found in '{suite_dir.relative_to(ROOT)}/'. Nothing to zip.")
        sys.exit(0)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"CREATING CONSOLIDATED ARCHIVE: {zip_path.name}")
    print(f"Mode:          {mode_label}")
    print(f"Source Folder: {suite_dir.relative_to(ROOT)}/ ({len(files_to_zip)} file(s))")
    print(f"Destination:   {zip_path.relative_to(ROOT)}")
    print("=" * 80)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, rel_path in files_to_zip:
            arcname = f"{suite.name}/{rel_path}"
            zf.write(file_path, arcname=str(arcname))
            print(f"  [+] Compressed: {arcname}")

        raw_csv = OUTPUT_DIR / "raw_all.csv"
        if raw_csv.exists():
            zf.write(raw_csv, arcname="raw_all.csv")
            print("  [+] Compressed: raw_all.csv")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f"SUCCESS: Package created -> {zip_path.name} ({size_mb:.2f} MB)")
    print(f"Location: {zip_path}")
    print("=" * 80)


def resolve_selection(suite: Suite, query: str) -> list[SuiteStep]:
    tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
    selected = []
    for token in tokens:
        match = None
        for s in suite.steps:
            code_lower = s.codename.lower()
            folder_lower = s.folder.lower()
            # 1. Exact match on numeric ID (e.g. "1", "2")
            if str(s.id) == token:
                match = s
                break
            # 2. Exact match on codename or folder name (e.g. "1a_dist-clip-ft")
            if code_lower == token or folder_lower == token:
                match = s
                break
            # 3. Match on prefix (e.g. "1a", "2b")
            if (
                folder_lower.startswith(f"{token}_")
                or code_lower.startswith(f"{token}_")
                or code_lower.startswith(f"{token}-")
            ):
                match = s
                break
            # 4. Match on name without index prefix (e.g. "dist-clip-ft" matching "1A_dist-clip-ft")
            clean_code = code_lower.split("_", 1)[-1] if "_" in code_lower else code_lower
            clean_folder = folder_lower.split("_", 1)[-1] if "_" in folder_lower else folder_lower
            if clean_code == token or clean_folder == token:
                match = s
                break
        if match:
            if match not in selected:
                selected.append(match)
        else:
            valid_keys = [f"{s.id} ({s.codename})" for s in suite.steps]
            print(f"Error: Unknown step or codename '{token}'. Valid options in suite '{suite.name}' are: {', '.join(valid_keys)}")
            sys.exit(1)
    return selected


def play_sound(kind: str = "step_done"):
    """Play a non-blocking system audio notification on completion/error."""
    import platform
    if platform.system() == "Darwin":
        sound_map = {
            "step_done": "/System/Library/Sounds/Glass.aiff",
            "suite_done": "/System/Library/Sounds/Hero.aiff",
            "error": "/System/Library/Sounds/Basso.aiff"
        }
        sound_path = sound_map.get(kind, "/System/Library/Sounds/Glass.aiff")
        try:
            subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    # Fallback to terminal bell
    sys.stdout.write("\a")
    sys.stdout.flush()


def run_suite_steps(suite: Suite, steps_to_run: list[SuiteStep]):
    suite_dir = suite.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    names = [s.codename for s in steps_to_run]
    print("=" * 80)
    print(f"EXECUTING SUITE: '{suite.name}' ({suite.title})")
    print(f"SELECTED STEPS ({len(steps_to_run)}): {', '.join(names)}")
    print(f"OUTPUT DESTINATION: {suite_dir.relative_to(ROOT)}/")
    print("=" * 80)

    for idx, step in enumerate(steps_to_run, 1):
        target_dir = suite_dir / step.folder
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(steps_to_run)}] >>> [{suite.name}/{step.folder}] {step.title}")
        print(f"Executing: {' '.join(step.cmd)}")

        before_snap = snapshot_output(suite.name)
        step_start = time.time()

        res = subprocess.run(step.cmd, cwd=ROOT)
        if res.returncode != 0:
            play_sound("error")
            print(f"\n[ERROR] Step '{step.codename}' failed with exit code {res.returncode}")
            sys.exit(res.returncode)

        step_elapsed = time.time() - step_start
        print(f"\n[{step.codename}] Finished in {step_elapsed:.1f}s. Detecting generated outputs...")

        after_snap = snapshot_output(suite.name)
        changed_files = []
        for p, mtime in after_snap.items():
            if p.name == "raw_all.csv" or p.name.startswith(".") or p.name == ".DS_Store":
                continue
            if p not in before_snap or mtime > before_snap[p]:
                changed_files.append(p)

        copied_count = 0
        for src_file in changed_files:
            dest = target_dir / src_file.name
            shutil.copy2(src_file, dest)
            copied_count += 1
            print(f"  [+] Output Saved: {dest.relative_to(ROOT)}")

        print(f"[{suite.name}/{step.folder}] {copied_count} output file(s) stored in {target_dir.relative_to(ROOT)}")
        play_sound("step_done")

    total_elapsed = time.time() - total_start
    play_sound("suite_done")
    print("\n" + "=" * 80)
    print(f"SUITE '{suite.name}' RUN(S) COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Outputs stored in: {suite_dir.relative_to(ROOT)}/")
    print(f"To package all outputs into a zip archive, run: uv run run_{suite.name}.py --zip")
    print("=" * 80)


def run_suite_cli(suite: Suite):
    """Universal CLI handler for any suite runner."""
    ap = argparse.ArgumentParser(description=f"Experiment Runner for Suite '{suite.name}'")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help=f"Run all {len(suite.steps)} experiments in this suite")
    group.add_argument("--run", "-r", type=str, help="Sub-run codename(s) or number(s) to run (e.g. '1,3' or 'joint-c')")
    group.add_argument("--list", "-l", action="store_true", help="List all codenames, status, and commands in this suite")
    group.add_argument("--zip", "-z", action="store_true", help="Package all completed outputs of this suite into a zip file")
    group.add_argument("--sync", "-s", action="store_true", help="Download and synchronize completed outputs from Modal Cloud Volume")

    ap.add_argument("--modal", "-m", action="store_true", help="Execute on Modal Serverless (16-core CPU containers in cloud)")
    ap.add_argument("--detach", "-d", action="store_true", help="Execute in detached background mode on Modal (safe from connection drops)")
    args = ap.parse_args()

    if args.list:
        render_suite_table(suite)
        return

    if args.sync:
        from src.utils.modal_engine import sync_modal_outputs
        sync_modal_outputs(suite)
        return

    if args.zip:
        package_suite_zip(suite, from_modal=args.modal)
        return

    if args.all:
        selected = suite.steps
    else:
        selected = resolve_selection(suite, args.run)

    if args.modal or args.detach:
        from src.utils.modal_engine import run_suite_steps_modal
        run_suite_steps_modal(suite, selected, detach=args.detach)
    else:
        run_suite_steps(suite, selected)

