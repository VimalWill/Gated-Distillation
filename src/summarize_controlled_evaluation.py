"""Summarize metrics and lm-eval completion for a controlled evaluation suite."""
import argparse
import json
from pathlib import Path


MODELS = ("pythia-160m", "Llama-3.2-1B", "gpt2")
COPIES = (1, 5, 10)


def load_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Controlled run directory, e.g. controlled-3313516")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    eval_root = run_dir / "evaluation-full"
    if not eval_root.is_dir():
        parser.error(f"Evaluation output not found: {eval_root}")

    rows = []
    missing = []
    for model in MODELS:
        for copies in COPIES:
            result_dir = eval_root / "runs" / model / f"copies-{copies}" / "seed-42"
            metrics_path = result_dir / "baseline" / "evaluation" / "metrics.json"
            if not metrics_path.is_file():
                missing.append(f"{model} copies={copies}: missing {metrics_path}")
                continue
            metrics = load_json(metrics_path)
            attack = metrics.get("attack", {})
            utility = metrics.get("utility", {})
            extraction = metrics.get("extraction", {})
            downstream_path = result_dir / "baseline" / "downstream_details.json"
            if downstream_path.is_file():
                downstream = load_json(downstream_path)
                lm_status = downstream.get("status", "unknown")
                sample_counts = [run.get("n_samples", {}) for run in downstream.get("runs", [])]
            else:
                lm_status, sample_counts = "not run", []
            rows.append({
                "model": model,
                "copies": copies,
                "n_eval": attack.get("n"),
                "members": attack.get("n_members"),
                "nonmembers": attack.get("n_nonmembers"),
                "raw_auc": attack.get("raw_auc"),
                "raw_auc_ci95": attack.get("ci95", {}).get("raw_auc"),
                "max_auc": attack.get("max_auc"),
                "utility_perplexity": utility.get("perplexity"),
                "utility_tokens": utility.get("n_tokens"),
                "exact_extraction": extraction.get("exact_match"),
                "token_recovery": extraction.get("token_recovery"),
                "lm_eval_status": lm_status,
                "lm_eval_sample_counts": sample_counts,
            })

    output_json = eval_root / "summary.json"
    output_md = eval_root / "summary.md"
    payload = {"run_dir": str(run_dir), "evaluations": rows, "missing": missing}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    headings = ("Model", "Copies", "N", "Member/Nonmember", "Raw AUC [95% CI]",
                "Max AUC", "Utility PPL", "Utility tokens", "Exact extraction",
                "Token recovery", "lm-eval")
    lines = ["| " + " | ".join(headings) + " |",
             "|" + "|".join("---" for _ in headings) + "|"]
    for row in rows:
        ci = row["raw_auc_ci95"]
        auc = "N/A" if row["raw_auc"] is None else f"{row['raw_auc']:.4f}"
        if ci:
            auc += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
        pair = f"{row['members']}/{row['nonmembers']}" if row["members"] is not None else "N/A"
        vals = (row["model"], str(row["copies"]), str(row["n_eval"] or "N/A"), pair,
                auc,
                "N/A" if row["max_auc"] is None else f"{row['max_auc']:.4f}",
                "N/A" if row["utility_perplexity"] is None else f"{row['utility_perplexity']:.4f}",
                str(row["utility_tokens"] or "N/A"),
                "N/A" if row["exact_extraction"] is None else f"{row['exact_extraction']:.4f}",
                "N/A" if row["token_recovery"] is None else f"{row['token_recovery']:.4f}",
                row["lm_eval_status"])
        lines.append("| " + " | ".join(vals) + " |")
    if missing:
        lines.extend(("", "Missing evaluations:", *[f"- {message}" for message in missing]))
    output_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {output_md}")
    print(f"Saved: {output_json}")
    if len(rows) != len(MODELS) * len(COPIES):
        raise SystemExit(f"Only {len(rows)} of 9 evaluation summaries were found")


if __name__ == "__main__":
    main()
