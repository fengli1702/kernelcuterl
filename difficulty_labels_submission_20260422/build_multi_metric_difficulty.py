#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


LEVEL_LABEL = {
    1: "level1_easy",
    2: "level2_medium",
    3: "level3_hard",
    4: "level4_expert",
}


def parse_bool(v):
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def parse_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def q123(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(x) for x in np.percentile(arr, [25, 50, 75]))


def level_by_quantile(x: float, q1: float, q2: float, q3: float, higher_harder: bool = True) -> int:
    if not np.isfinite(x):
        return 2
    if higher_harder:
        if x <= q1:
            return 1
        if x <= q2:
            return 2
        if x <= q3:
            return 3
        return 4
    # lower is harder (e.g. speedup)
    if x <= q1:
        return 4
    if x <= q2:
        return 3
    if x <= q3:
        return 2
    return 1


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return {k: float(v) / total for k, v in raw_weights.items()}


def main():
    parser = argparse.ArgumentParser(description="Build per-metric and weighted difficulty grades.")
    parser.add_argument("--per-task", default="/home/i_lidaifeng/qwen/per_task_report.csv")
    parser.add_argument("--ir-metrics", default="/home/i_lidaifeng/qwen/model_ops_ir_torch_graph_metrics_rerun.csv")
    parser.add_argument("--out-csv", default="/home/i_lidaifeng/qwen/per_task_multi_metric_difficulty.csv")
    parser.add_argument("--out-json", default="/home/i_lidaifeng/qwen/per_task_multi_metric_difficulty_summary.json")
    parser.add_argument("--out-report", default="/home/i_lidaifeng/qwen/PER_TASK_MULTI_METRIC_DIFFICULTY_REPORT.md")

    parser.add_argument("--w-ops", type=float, default=0.25)
    parser.add_argument("--w-ast-depth", type=float, default=0.25)
    parser.add_argument("--w-output-len", type=float, default=0.15)
    parser.add_argument("--w-correctness", type=float, default=0.20)
    parser.add_argument("--w-speedup", type=float, default=0.15)

    args = parser.parse_args()

    raw_weights = {
        "ops": args.w_ops,
        "ast_depth": args.w_ast_depth,
        "output_len": args.w_output_len,
        "correctness": args.w_correctness,
        "speedup": args.w_speedup,
    }
    weights = normalize_weights(raw_weights)

    per_rows = list(csv.DictReader(open(args.per_task, newline="", encoding="utf-8")))
    ir_rows = list(csv.DictReader(open(args.ir_metrics, newline="", encoding="utf-8")))
    ir_map = {r["task_id"]: r for r in ir_rows}

    merged = []
    for r in per_rows:
        task_id = r["task_id"]
        ir = ir_map.get(task_id, {})

        ops_from_ir = parse_float(ir.get("model_ops_total_calls_v3"))
        ops_fallback = parse_float(r.get("ops_count"))
        ops_val = ops_from_ir if np.isfinite(ops_from_ir) else ops_fallback

        fg_depth = parse_float(ir.get("forward_graph_depth_v3"))
        ast_fg_depth = parse_float(ir.get("ast_forward_graph_depth"))
        model_ast_depth = parse_float(ir.get("model_ast_depth"))

        if np.isfinite(fg_depth):
            ast_depth_val = fg_depth
            ast_depth_source = "forward_graph_depth_v3"
        elif np.isfinite(ast_fg_depth):
            ast_depth_val = ast_fg_depth
            ast_depth_source = "ast_forward_graph_depth"
        else:
            ast_depth_val = model_ast_depth
            ast_depth_source = "model_ast_depth"

        output_len = parse_float(r.get("output_tokens"))
        speedup = parse_float(r.get("speedup"))
        compiled = parse_bool(r.get("compiled"))
        correct = parse_bool(r.get("correct"))

        merged.append(
            {
                **r,
                "ir_source": ir.get("ir_source", ""),
                "trace_status": ir.get("trace_status", ""),
                "ops_metric": ops_val,
                "ast_depth_metric": ast_depth_val,
                "ast_depth_source": ast_depth_source,
                "output_len_metric": output_len,
                "speedup_metric": speedup,
                "compiled_bool": compiled,
                "correct_bool": correct,
            }
        )

    ops_q = q123([x["ops_metric"] for x in merged])
    ast_q = q123([x["ast_depth_metric"] for x in merged])
    out_q = q123([x["output_len_metric"] for x in merged])
    spd_q = q123([x["speedup_metric"] for x in merged])

    for x in merged:
        x["ops_level"] = level_by_quantile(x["ops_metric"], *ops_q, higher_harder=True)
        x["ast_depth_level"] = level_by_quantile(x["ast_depth_metric"], *ast_q, higher_harder=True)
        x["output_len_level"] = level_by_quantile(x["output_len_metric"], *out_q, higher_harder=True)
        x["speedup_level"] = level_by_quantile(x["speedup_metric"], *spd_q, higher_harder=False)

        # correctness metric is a one-shot outcome metric:
        # correct=True -> easy, compiled=True&correct=False -> hard, compiled=False -> expert
        if x["compiled_bool"] is False:
            x["correctness_level"] = 4
        elif x["correct_bool"] is False:
            x["correctness_level"] = 3
        else:
            x["correctness_level"] = 1

        x["weighted_difficulty_score"] = (
            weights["ops"] * x["ops_level"]
            + weights["ast_depth"] * x["ast_depth_level"]
            + weights["output_len"] * x["output_len_level"]
            + weights["correctness"] * x["correctness_level"]
            + weights["speedup"] * x["speedup_level"]
        )

    weighted_scores = [x["weighted_difficulty_score"] for x in merged]
    final_q = q123(weighted_scores)
    for x in merged:
        x["final_level"] = level_by_quantile(x["weighted_difficulty_score"], *final_q, higher_harder=True)
        x["final_label"] = LEVEL_LABEL[x["final_level"]]
        x["ops_level_label"] = LEVEL_LABEL[x["ops_level"]]
        x["ast_depth_level_label"] = LEVEL_LABEL[x["ast_depth_level"]]
        x["output_len_level_label"] = LEVEL_LABEL[x["output_len_level"]]
        x["correctness_level_label"] = LEVEL_LABEL[x["correctness_level"]]
        x["speedup_level_label"] = LEVEL_LABEL[x["speedup_level"]]

    # save csv
    fieldnames = [
        "task_id",
        "compiled",
        "correct",
        "speedup",
        "error",
        "input_tokens",
        "output_tokens",
        "content_length",
        "ops_count",
        "difficulty",
        "difficulty_level",
        "ops_group",
        "ir_source",
        "trace_status",
        "ops_metric",
        "ast_depth_metric",
        "ast_depth_source",
        "output_len_metric",
        "speedup_metric",
        "ops_level",
        "ops_level_label",
        "ast_depth_level",
        "ast_depth_level_label",
        "output_len_level",
        "output_len_level_label",
        "correctness_level",
        "correctness_level_label",
        "speedup_level",
        "speedup_level_label",
        "weighted_difficulty_score",
        "final_level",
        "final_label",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for x in merged:
            row = {k: x.get(k, "") for k in fieldnames}
            w.writerow(row)

    # summaries
    def level_dist(key):
        c = Counter(int(x[key]) for x in merged)
        return {f"level{lv}": int(c.get(lv, 0)) for lv in [1, 2, 3, 4]}

    def avg_by_final(key):
        out = {}
        for lv in [1, 2, 3, 4]:
            vals = [x[key] for x in merged if int(x["final_level"]) == lv and np.isfinite(x[key])]
            if not vals:
                out[f"level{lv}"] = {"count": 0, "mean": None, "median": None}
            else:
                out[f"level{lv}"] = {
                    "count": len(vals),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                }
        return out

    summary = {
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "per_task": args.per_task,
            "ir_metrics": args.ir_metrics,
            "rows": len(merged),
        },
        "weights_normalized": weights,
        "metric_thresholds": {
            "ops_q25_q50_q75": ops_q,
            "ast_depth_q25_q50_q75": ast_q,
            "output_len_q25_q50_q75": out_q,
            "speedup_q25_q50_q75": spd_q,
            "final_weighted_score_q25_q50_q75": final_q,
        },
        "per_metric_level_distribution": {
            "ops_level": level_dist("ops_level"),
            "ast_depth_level": level_dist("ast_depth_level"),
            "output_len_level": level_dist("output_len_level"),
            "correctness_level": level_dist("correctness_level"),
            "speedup_level": level_dist("speedup_level"),
        },
        "final_level_distribution": level_dist("final_level"),
        "metric_mean_by_final_level": {
            "ops_metric": avg_by_final("ops_metric"),
            "ast_depth_metric": avg_by_final("ast_depth_metric"),
            "output_len_metric": avg_by_final("output_len_metric"),
            "speedup_metric": avg_by_final("speedup_metric"),
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # markdown report
    def pct(v):
        return f"{100.0 * v / len(merged):.2f}%"

    md = []
    md.append("# 多指标难度分级报告")
    md.append("")
    md.append("## 指标与方法")
    md.append("")
    md.append("- 指标1 `ops_metric`: `model_ops_total_calls_v3`（缺失回退 `ops_count`），值越大越难")
    md.append("- 指标2 `ast_depth_metric`: 优先 `forward_graph_depth_v3`，再回退 `ast_forward_graph_depth`，值越大越难")
    md.append("- 指标3 `output_len_metric`: `output_tokens`，值越大越难")
    md.append("- 指标4 `correctness_level`: 一次运行口径（correct=True→L1；compiled=True且correct=False→L3；compiled=False→L4）")
    md.append("- 指标5 `speedup_level`: `speedup`，值越小越难")
    md.append("- 连续指标均按全量Q25/Q50/Q75分成4档，最后做加权汇总后再按Q25/Q50/Q75分最终4档")
    md.append("")
    md.append("## 权重")
    md.append("")
    for k in ["ops", "ast_depth", "output_len", "correctness", "speedup"]:
        md.append(f"- `{k}`: {weights[k]:.4f}")
    md.append("")
    md.append("## 阈值")
    md.append("")
    md.append(f"- ops q25/q50/q75: {ops_q}")
    md.append(f"- ast_depth q25/q50/q75: {ast_q}")
    md.append(f"- output_len q25/q50/q75: {out_q}")
    md.append(f"- speedup q25/q50/q75: {spd_q}")
    md.append(f"- final weighted score q25/q50/q75: {final_q}")
    md.append("")

    md.append("## 各指标分级分布")
    md.append("")
    md.append("| 指标 | L1 | L2 | L3 | L4 |")
    md.append("|---|---:|---:|---:|---:|")
    for key in ["ops_level", "ast_depth_level", "output_len_level", "correctness_level", "speedup_level"]:
        dist = summary["per_metric_level_distribution"][key]
        md.append(
            f"| {key} | {dist['level1']} ({pct(dist['level1'])}) | {dist['level2']} ({pct(dist['level2'])}) | {dist['level3']} ({pct(dist['level3'])}) | {dist['level4']} ({pct(dist['level4'])}) |"
        )
    md.append("")

    md.append("## 加权汇总后最终分级")
    md.append("")
    final_dist = summary["final_level_distribution"]
    md.append("| 最终等级 | 数量 | 占比 |")
    md.append("|---|---:|---:|")
    for lv in [1, 2, 3, 4]:
        c = final_dist[f"level{lv}"]
        md.append(f"| {LEVEL_LABEL[lv]} | {c} | {pct(c)} |")
    md.append("")

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("written_csv", args.out_csv)
    print("written_json", args.out_json)
    print("written_report", args.out_report)


if __name__ == "__main__":
    main()
