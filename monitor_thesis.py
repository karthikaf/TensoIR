"""
Thesis Hardware & Training Monitor
Logs GPU, CPU, RAM, Disk I/O, and training progress at regular intervals.
Output: monitor_log.csv  (machine-readable)
        monitor_log.txt  (human-readable summary)
"""

import time, csv, os, subprocess, re, datetime, psutil, json

LOG_CSV  = "/root/TensoIR/thesis_monitor_log.csv"
LOG_TXT  = "/root/TensoIR/thesis_monitor_log.txt"
INTERVAL = 60  # seconds between samples
LOG_DIR  = "/root/TensoIR/log/log_rotated_multi_lights"

CSV_FIELDS = [
    "timestamp", "elapsed_min",
    # GPU
    "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_pct",
    "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w",
    # CPU / RAM
    "cpu_util_pct", "ram_used_gb", "ram_total_gb", "ram_pct",
    # Disk I/O (cumulative bytes since boot)
    "disk_read_gb", "disk_write_gb",
    # Network (cumulative bytes since boot)
    "net_sent_gb", "net_recv_gb",
    # Training progress (latest iter found across all active runs)
    "active_runs", "ficus_stage", "ficus_iter", "ficus_psnr_train",
    "hotdog_stage", "hotdog_iter", "hotdog_psnr_train",
    "lego_stage",  "lego_iter",  "lego_psnr_train",
]

def nvidia_smi():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits"
        ]).decode().strip().split(",")
        return {
            "gpu_util_pct":      float(out[0].strip()),
            "gpu_mem_used_mb":   float(out[1].strip()),
            "gpu_mem_total_mb":  float(out[2].strip()),
            "gpu_mem_pct":       round(float(out[1].strip()) / float(out[2].strip()) * 100, 2),
            "gpu_temp_c":        float(out[3].strip()),
            "gpu_power_w":       float(out[4].strip()),
            "gpu_power_limit_w": float(out[5].strip()),
        }
    except Exception as e:
        return {k: None for k in ["gpu_util_pct","gpu_mem_used_mb","gpu_mem_total_mb",
                                   "gpu_mem_pct","gpu_temp_c","gpu_power_w","gpu_power_limit_w"]}

def get_training_progress(scene):
    """Scan tmux pane for latest iter/PSNR for a given scene.
    Tries multiple session name conventions: {scene}_v3, {scene}_v2,
    {scene}_runs, {scene}, and any session containing the scene name.
    """
    # Build candidate session names to try in order
    candidates = [f"{scene}_v3", f"{scene}_v2", f"{scene}_runs", scene]
    # Also check any live tmux session whose name contains the scene name
    try:
        sessions = subprocess.check_output(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")
        for s in sessions:
            if scene in s and s not in candidates:
                candidates.append(s)
    except Exception:
        pass

    for target in candidates:
        try:
            pane = subprocess.check_output(
                ["tmux", "capture-pane", "-t", target, "-p"],
                stderr=subprocess.DEVNULL
            ).decode()
            iters = re.findall(r"Iteration\s+(\d+)\s+PSNR:\s+train_rgb\s*=\s*([\d.]+)", pane)
            if iters:
                it, psnr = iters[-1]
                stage = "stage2" if "stage2" in pane.split("Iteration")[0][-2000:] else \
                        "stage1" if "stage1" in pane.split("Iteration")[0][-2000:] else "vanilla"
                return stage, int(it), float(psnr)
        except Exception:
            continue
    return "unknown", 0, 0.0

def count_active_runs():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]).decode()
        return len([l for l in out.strip().split("\n") if l.strip()])
    except:
        return 0

def human_line(row):
    ts  = row["timestamp"]
    ela = row["elapsed_min"]
    return (
        f"[{ts}] +{ela:.0f}min | "
        f"GPU {row['gpu_util_pct']}% {row['gpu_mem_used_mb']:.0f}/{row['gpu_mem_total_mb']:.0f}MB "
        f"({row['gpu_mem_pct']}%) {row['gpu_temp_c']}°C {row['gpu_power_w']}W/{row['gpu_power_limit_w']}W | "
        f"CPU {row['cpu_util_pct']}% RAM {row['ram_used_gb']:.1f}/{row['ram_total_gb']:.1f}GB | "
        f"Net ↑{row['net_sent_gb']:.2f}GB ↓{row['net_recv_gb']:.2f}GB | "
        f"Disk R:{row['disk_read_gb']:.2f}GB W:{row['disk_write_gb']:.2f}GB | "
        f"Active procs: {row['active_runs']} | "
        f"ficus({row['ficus_stage']} it={row['ficus_iter']} psnr={row['ficus_psnr_train']}) "
        f"hotdog({row['hotdog_stage']} it={row['hotdog_iter']} psnr={row['hotdog_psnr_train']}) "
        f"lego({row['lego_stage']} it={row['lego_iter']} psnr={row['lego_psnr_train']})"
    )

def main():
    start = time.time()
    write_header = not os.path.exists(LOG_CSV)

    print(f"Thesis monitor started. Logging every {INTERVAL}s.")
    print(f"  CSV : {LOG_CSV}")
    print(f"  TXT : {LOG_TXT}")

    with open(LOG_CSV, "a", newline="") as cf, open(LOG_TXT, "a") as tf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            tf.write("="*120 + "\n")
            tf.write(f"Thesis Monitor Started: {datetime.datetime.now().isoformat()}\n")
            tf.write("="*120 + "\n\n")

        while True:
            now     = datetime.datetime.now()
            elapsed = (time.time() - start) / 60

            gpu   = nvidia_smi()
            cpu   = psutil.cpu_percent(interval=1)
            ram   = psutil.virtual_memory()
            disk  = psutil.disk_io_counters()
            net   = psutil.net_io_counters()

            f_stage, f_iter, f_psnr = get_training_progress("ficus")
            h_stage, h_iter, h_psnr = get_training_progress("hotdog")
            l_stage, l_iter, l_psnr = get_training_progress("lego")

            row = {
                "timestamp":       now.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_min":     round(elapsed, 2),
                "gpu_util_pct":    gpu["gpu_util_pct"],
                "gpu_mem_used_mb": gpu["gpu_mem_used_mb"],
                "gpu_mem_total_mb":gpu["gpu_mem_total_mb"],
                "gpu_mem_pct":     gpu["gpu_mem_pct"],
                "gpu_temp_c":      gpu["gpu_temp_c"],
                "gpu_power_w":     gpu["gpu_power_w"],
                "gpu_power_limit_w": gpu["gpu_power_limit_w"],
                "cpu_util_pct":    cpu,
                "ram_used_gb":     round(ram.used / 1e9, 2),
                "ram_total_gb":    round(ram.total / 1e9, 2),
                "ram_pct":         ram.percent,
                "disk_read_gb":    round(disk.read_bytes / 1e9, 3),
                "disk_write_gb":   round(disk.write_bytes / 1e9, 3),
                "net_sent_gb":     round(net.bytes_sent / 1e9, 3),
                "net_recv_gb":     round(net.bytes_recv / 1e9, 3),
                "active_runs":     count_active_runs(),
                "ficus_stage":     f_stage, "ficus_iter": f_iter, "ficus_psnr_train": f_psnr,
                "hotdog_stage":    h_stage, "hotdog_iter": h_iter, "hotdog_psnr_train": h_psnr,
                "lego_stage":      l_stage, "lego_iter":  l_iter,  "lego_psnr_train":  l_psnr,
            }

            writer.writerow(row)
            cf.flush()

            line = human_line(row)
            tf.write(line + "\n")
            tf.flush()
            print(line)

            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
