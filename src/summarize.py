import argparse, os, glob, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--carbon_kg_per_kwh", type=float, default=0.4)
    ap.add_argument("--out", default="./results/summary.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "total_latency_ms" not in df.columns:
            continue
        n = len(df)
        energy_j = float(df.get("energy_cpu_pkg_j", pd.Series([0]*n)).fillna(0).sum())
        total_energy_wh = energy_j/3600.0
        energy_per_log_wh = total_energy_wh/n if n else 0.0
        avg_lat = float(df["total_latency_ms"].mean())
        p50_lat = float(df["total_latency_ms"].quantile(0.5))
        tot_time_s = float(df["total_latency_ms"].sum())/1000.0
        thr = n/tot_time_s if tot_time_s > 0 else 0.0
        carbon = (total_energy_wh/1000.0) * args.carbon_kg_per_kwh
        succ = float(df["success"].mean()) if "success" in df.columns else None

        fname = os.path.basename(path)
        method = fname.split("_", 1)[0]
        dataset = fname.split("_", 1)[1].replace(".csv", "") if "_" in fname else "unknown"

        rows.append(dict(
            Method=method, Dataset=dataset,
            LogsProcessed=int(n),
            TotalEnergyWh=total_energy_wh,
            EnergyPerLogWh=energy_per_log_wh,
            CarbonEmissionsKg=carbon,
            AvgLatencyMs=avg_lat,
            P50LatencyMs=p50_lat,
            ThroughputLogsPerSec=thr,
            SuccessRate=succ
        ))

    out = pd.DataFrame(rows).sort_values(["Dataset","Method"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
