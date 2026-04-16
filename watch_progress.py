#!/usr/bin/env python3
"""Live progress bar display for TensoIR training runs."""
import re, time
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich import box

LOG_FILE = "/root/TensoIR/thesis_monitor_log.txt"

RUNS = {
    "ficus":  {"label": "A2 Ficus  Stage 1", "total": 40_000},
    "hotdog": {"label": "B2 Hotdog Stage 1", "total": 40_000},
    "lego":   {"label": "C2 Lego   Stage 1", "total": 40_000},
}

def parse_last_log():
    try:
        with open(LOG_FILE) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None
        last = lines[-1]
        # timestamp
        ts_m = re.search(r'\[(.+?)\]', last)
        ts = ts_m.group(1) if ts_m else "—"
        # GPU
        gpu_m = re.search(r'GPU (\d+\.?\d*)%\s+([\d]+)/([\d]+)MB.*?(\d+\.?\d*)W/(\d+\.?\d*)W', last)
        gpu = gpu_m.groups() if gpu_m else None
        # per-scene iters and psnr
        scenes = {}
        for scene in RUNS:
            m = re.search(rf'{scene}\(\S+ it=(\d+) psnr=([\d.]+)\)', last)
            if m:
                scenes[scene] = {"it": int(m.group(1)), "psnr": float(m.group(2))}
        return {"ts": ts, "gpu": gpu, "scenes": scenes}
    except Exception:
        return None

def make_table(data):
    if not data:
        return Panel("[yellow]Waiting for log data...[/yellow]")

    gpu = data["gpu"]
    gpu_str = f"GPU {gpu[0]}%  {int(gpu[1])//1024}/{int(gpu[2])//1024} GB  {float(gpu[3]):.0f}/{float(gpu[4]):.0f} W" if gpu else "—"

    table = Table(
        title=f"[bold cyan]TensoIR Training — {data['ts']}[/bold cyan]\n[dim]{gpu_str}[/dim]",
        box=box.ROUNDED, show_header=True, header_style="bold white",
        min_width=78
    )
    table.add_column("Run",    style="bold", width=20)
    table.add_column("Progress", width=30)
    table.add_column("Iters",  justify="right", width=12)
    table.add_column("PSNR",   justify="right", width=8)
    table.add_column("ETA",    justify="right", width=10)

    for scene, meta in RUNS.items():
        info = data["scenes"].get(scene)
        total = meta["total"]
        label = meta["label"]

        if info:
            it = min(info["it"], total)
            psnr = info["psnr"]
            pct = it / total
            done = round(pct * 28)
            bar = f"[green]{'█' * done}[/green][dim]{'░' * (28 - done)}[/dim] {pct*100:5.1f}%"
            # rough ETA: assume linear from current rate (use elapsed since start would be better,
            # but we only have current iter — use a fixed ~400 iters/min across 3 procs)
            iters_left = total - it
            eta_min = iters_left / 130  # ~130 iters/min per process (400 total / 3)
            if eta_min > 90:
                eta_str = f"{eta_min/60:.1f}h"
            else:
                eta_str = f"{eta_min:.0f}m"
            psnr_str = f"{psnr:.2f}"
            iter_str = f"{it:,}/{total:,}"
        else:
            bar = "[dim]░" * 28 + "[/dim]  —"
            eta_str = "—"
            psnr_str = "—"
            iter_str = "—"

        if info and info["it"] >= total:
            label = f"[green]✅ {label}[/green]"
            eta_str = "done"
        else:
            label = f"[cyan]▶ {label}[/cyan]"

        table.add_row(label, bar, iter_str, psnr_str, eta_str)

    return Panel(table, border_style="bright_black")

console = Console()
console.print("[bold green]Watching training runs — Ctrl+C to exit[/bold green]\n")

with Live(make_table(parse_last_log()), refresh_per_second=0.5, console=console) as live:
    try:
        while True:
            data = parse_last_log()
            live.update(make_table(data))
            time.sleep(15)
    except KeyboardInterrupt:
        pass

console.print("\n[dim]Stopped.[/dim]")
