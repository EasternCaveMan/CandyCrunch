from pathlib import Path


def _read_kib(pid):
    values = {
        "Rss": 0,
        "Pss": 0,
        "Pss_Anon": 0,
        "Pss_File": 0,
        "Pss_Shmem": 0,
        "Private_Clean": 0,
        "Private_Dirty": 0,
        "SwapPss": 0,
    }
    try:
        with Path(f"/proc/{pid}/smaps_rollup").open() as file:
            for line in file:
                key = line.split(":", 1)[0]
                if key in values:
                    values[key] = int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return values


def _descendants(pid):
    pending = [pid]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pending.extend(int(child) for child in children.split())
    return seen


def report_process_tree_memory(stage):
    import os

    totals = {
        "Rss": 0,
        "Pss": 0,
        "Pss_Anon": 0,
        "Pss_File": 0,
        "Pss_Shmem": 0,
        "Private_Clean": 0,
        "Private_Dirty": 0,
        "SwapPss": 0,
    }
    process_count = 0
    for pid in _descendants(os.getpid()):
        values = _read_kib(pid)
        if values is None:
            continue
        process_count += 1
        for key, value in values.items():
            totals[key] += value
    rss_gib = totals["Rss"] / 1024**2
    pss_gib = totals["Pss"] / 1024**2
    anon_gib = totals["Pss_Anon"] / 1024**2
    file_gib = totals["Pss_File"] / 1024**2
    uss_gib = (totals["Private_Clean"] + totals["Private_Dirty"]) / 1024**2
    swap_gib = totals["SwapPss"] / 1024**2
    print(
        f"[RAM] {stage}: processes={process_count} "
        f"RSS(sum)={rss_gib:.2f} GiB PSS(sum)={pss_gib:.2f} GiB "
        f"AnonPSS={anon_gib:.2f} GiB FilePSS={file_gib:.2f} GiB "
        f"USS(sum)={uss_gib:.2f} GiB SwapPSS={swap_gib:.2f} GiB"
    )
