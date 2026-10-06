import re
import ast
from collections import defaultdict
from pprint import pprint

EVAL_HEADER_RE = re.compile(
    r'^\[EVAL\]\s+Running with\s+'
    r'NUM_SAMPLES=(?P<num_samples>\d+),\s*'
    r'NUM_BATCHES=(?P<num_batches>\d+),\s*'
    r'T=(?P<temperature>[-+]?[\d.]+),\s*'
    r'MODE=(?P<mode>[A-Za-z0-9_]+)\s*$',
    re.MULTILINE
)

def _find_metrics_dict(block: str) -> dict:
    """
    Find the first Python dict literal in the given text block and return it as a dict.
    Assumes the dict uses Python literal syntax (single quotes ok), which ast.literal_eval can parse.
    """
    m = re.search(r'\{.*?\}', block, flags=re.DOTALL)
    if not m:
        raise ValueError("No metrics dict found in a block.")
    return ast.literal_eval(m.group(0))

def parse_eval_log(text: str):
    """
    Parse the evaluation log text and return structured data.
    Returns:
      runs: list of dicts with keys:
            - num_samples (int)
            - num_batches (int)
            - temperature (float)
            - mode (str)
            - metrics (dict)
      by_mode_then_temp: dict like {mode: {temperature: metrics_dict}}
      table_rows: list of flat dicts (good for a DataFrame later)
    """
    runs = []
    matches = list(EVAL_HEADER_RE.finditer(text))

    for i, hdr in enumerate(matches):
        start = hdr.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        metrics = _find_metrics_dict(block)

        run = {
            "num_samples": int(hdr.group("num_samples")),
            "num_batches": int(hdr.group("num_batches")),
            "temperature": float(hdr.group("temperature")),
            "mode": hdr.group("mode"),
            "metrics": metrics,
        }
        runs.append(run)

    by_mode_then_temp = defaultdict(dict)
    for r in runs:
        by_mode_then_temp[r["mode"]][r["temperature"]] = r["metrics"]

    table_rows = []
    for r in runs:
        flat = {
            "mode": r["mode"],
            "temperature": r["temperature"],
            "num_samples_hdr": r["num_samples"],
            "num_batches_hdr": r["num_batches"],
        }
        for k, v in r["metrics"].items():
            flat[k] = v
        table_rows.append(flat)

    return runs, dict(by_mode_then_temp), table_rows

def read_text_safely(path: str) -> str:
    """
    Try UTF-8 and UTF-8 with BOM first, then fall back to cp1252 and latin-1.
    As a last resort, replace undecodable characters.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Last resort: replace undecodable bytes
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

if __name__ == "__main__":
    # Adjust this path to your log file
    LOG_PATH = "parsed_results.txt"
    LOG_TEXT = read_text_safely(LOG_PATH)

    runs, by_mode_then_temp, table_rows = parse_eval_log(LOG_TEXT)

    # Quick sanity prints; you can remove these
    # print(f"{len(runs)} runs parsed")
    print("Modes found:", sorted(by_mode_then_temp.keys()))
    print("Temperatures per mode:")
    for mode, temps in by_mode_then_temp.items():
        print(" ", mode, "->", sorted(temps.keys()))

    # Example: look up metrics for mode='topk' at T=1.0
    example = by_mode_then_temp.get("topk", {}).get(1.0)
    print("\nExample metrics for mode='topk', T=1.0:")

    topk_t0= by_mode_then_temp.get("topk").get(0.0)
    topp_t0 = by_mode_then_temp.get("topp").get(0.0)
    multi_t0 = by_mode_then_temp.get("multinomial").get(0.0)

    topp_t1 = by_mode_then_temp.get("topp").get(1.0)
    topk_t1 = by_mode_then_temp.get("topk").get(1.0)
    multi_t1 = by_mode_then_temp.get("multinomial").get(1.0)

    topp_t2 = by_mode_then_temp.get("topp").get(2.0)
    topk_t2 = by_mode_then_temp.get("topk").get(2.0)
    multi_t2 = by_mode_then_temp.get("multinomial").get(2.0)

    topp_t5 = by_mode_then_temp.get("topp").get(5.0)
    topk_t5 = by_mode_then_temp.get("topk").get(5.0)
    multi_t5 = by_mode_then_temp.get("multinomial").get(5.0)

    topp_t10 = by_mode_then_temp.get("topp").get(10.0)
    topk_t10 = by_mode_then_temp.get("topk").get(10.0)
    multi_t10 = by_mode_then_temp.get("multinomial").get(10.0)

    # print ADE_mean and FDE_mean for each mode and temperature
    print(f"topk T=0.0: ADE_mean={topk_t0['ADE_mean']}, FDE_mean={topk_t0['FDE_mean']}")
    print(f"topp T=0.0: ADE_mean={topp_t0['ADE_mean']}, FDE_mean={topp_t0['FDE_mean']}")
    print(f"multi T=0.0: ADE_mean={multi_t0['ADE_mean']}, FDE_mean={multi_t0['FDE_mean']}")
    print(f"topk T=1.0: ADE_mean={topk_t1['ADE_mean']}, FDE_mean={topk_t1['FDE_mean']}")
    print(f"topp T=1.0: ADE_mean={topp_t1['ADE_mean']}, FDE_mean={topp_t1['FDE_mean']}")
    print(f"multi T=1.0: ADE_mean={multi_t1['ADE_mean']}, FDE_mean={multi_t1['FDE_mean']}")
    print(f"topk T=2.0: ADE_mean={topk_t2['ADE_mean']}, FDE_mean={topk_t2['FDE_mean']}")
    print(f"topp T=2.0: ADE_mean={topp_t2['ADE_mean']}, FDE_mean={topp_t2['FDE_mean']}")
    print(f"multi T=2.0: ADE_mean={multi_t2['ADE_mean']}, FDE_mean={multi_t2['FDE_mean']}")
    print(f"topk T=5.0: ADE_mean={topk_t5['ADE_mean']}, FDE_mean={topk_t5['FDE_mean']}")
    print(f"topp T=5.0: ADE_mean={topp_t5['ADE_mean']}, FDE_mean={topp_t5['FDE_mean']}")
    print(f"multi T=5.0: ADE_mean={multi_t5['ADE_mean']}, FDE_mean={multi_t5['FDE_mean']}")
    print(f"topk T=10.0: ADE_mean={topk_t10['ADE_mean']}, FDE_mean={topk_t10['FDE_mean']}")
    print(f"topp T=10.0: ADE_mean={topp_t10['ADE_mean']}, FDE_mean={topp_t10['FDE_mean']}")
    print(f"multi T=10.0: ADE_mean={multi_t10['ADE_mean']}, FDE_mean={multi_t10['FDE_mean']}")

    # next min ade
    print("\nMinimum ADE across all modes and temperatures:")
    print(f"topk T=0.0: ADE_mean={topk_t0['minADE@5_mean']}, FDE_mean={topk_t0['minFDE@5_mean']}")
    print(f"topp T=0.0: ADE_mean={topp_t0['minADE@5_mean']}, FDE_mean={topp_t0['minFDE@5_mean']}")
    print(f"multi T=0.0: ADE_mean={multi_t0['minADE@5_mean']}, FDE_mean={multi_t0['minFDE@5_mean']}")

    print(f"topk T=1.0: ADE_mean={topk_t1['minADE@5_mean']}, FDE_mean={topk_t1['minFDE@5_mean']}")
    print(f"topp T=1.0: ADE_mean={topp_t1['minADE@5_mean']}, FDE_mean={topp_t1['minFDE@5_mean']}")
    print(f"multi T=1.0: ADE_mean={multi_t1['minADE@5_mean']}, FDE_mean={multi_t1['minFDE@5_mean']}")

    print(f"topk T=2.0: ADE_mean={topk_t2['minADE@5_mean']}, FDE_mean={topk_t2['minFDE@5_mean']}")
    print(f"topp T=2.0: ADE_mean={topp_t2['minADE@5_mean']}, FDE_mean={topp_t2['minFDE@5_mean']}")
    print(f"multi T=2.0: ADE_mean={multi_t2['minADE@5_mean']}, FDE_mean={multi_t2['minFDE@5_mean']}")

    print(f"topk T=5.0: ADE_mean={topk_t5['minADE@5_mean']}, FDE_mean={topk_t5['minFDE@5_mean']}")
    print(f"topp T=5.0: ADE_mean={topp_t5['minADE@5_mean']}, FDE_mean={topp_t5['minFDE@5_mean']}")
    print(f"multi T=5.0: ADE_mean={multi_t5['minADE@5_mean']}, FDE_mean={multi_t5['minFDE@5_mean']}")

    print(f"topk T=10.0: ADE_mean={topk_t10['minADE@5_mean']}, FDE_mean={topk_t10['minFDE@5_mean']}")
    print(f"topp T=10.0: ADE_mean={topp_t10['minADE@5_mean']}, FDE_mean={topp_t10['minFDE@5_mean']}")
    print(f"multi T=10.0: ADE_mean={multi_t10['minADE@5_mean']}, FDE_mean={multi_t10['minFDE@5_mean']}")

    div_mean_topk_t0 = topk_t0["diversity"]
    div_mean_topp_t0 = topp_t0["diversity"]
    div_mean_multi_t0 = multi_t0["diversity"]
    div_mean_topk_t1 = topk_t1["diversity"]
    div_mean_topp_t1 = topp_t1["diversity"]
    div_mean_multi_t1 = multi_t1["diversity"]
    div_mean_topk_t2 = topk_t2["diversity"]
    div_mean_topp_t2 = topp_t2["diversity"]
    div_mean_multi_t2 = multi_t2["diversity"]
    div_mean_topk_t5 = topk_t5["diversity"]
    div_mean_topp_t5 = topp_t5["diversity"]
    div_mean_multi_t5 = multi_t5["diversity"]
    div_mean_topk_t10 = topk_t10["diversity"]
    div_mean_topp_t10 = topp_t10["diversity"]
    div_mean_multi_t10 = multi_t10["diversity"]

    print("\nDiversity (L2 distance) across all modes and temperatures:")
    print(f"topk T=0.0: Diversity={div_mean_topk_t0}")
    print(f"topp T=0.0: Diversity={div_mean_topp_t0}")
    print(f"multi T=0.0: Diversity={div_mean_multi_t0}")
    print(f"topk T=1.0: Diversity={div_mean_topk_t1}")
    print(f"topp T=1.0: Diversity={div_mean_topp_t1}")
    print(f"multi T=1.0: Diversity={div_mean_multi_t1}")
    print(f"topk T=2.0: Diversity={div_mean_topk_t2}")
    print(f"topp T=2.0: Diversity={div_mean_topp_t2}")
    print(f"multi T=2.0: Diversity={div_mean_multi_t2}")
    print(f"topk T=5.0: Diversity={div_mean_topk_t5}")
    print(f"topp T=5.0: Diversity={div_mean_topp_t5}")
    print(f"multi T=5.0: Diversity={div_mean_multi_t5}")
    print(f"topk T=10.0: Diversity={div_mean_topk_t10}")
    print(f"topp T=10.0: Diversity={div_mean_topp_t10}")
    print(f"multi T=10.0: Diversity={div_mean_multi_t10}")

    div_topp_t0 = topp_t0["diversity_per_timestep"]
    div_topp_t1 = topp_t1["diversity_per_timestep"]
    div_topp_t2 = topp_t2["diversity_per_timestep"]
    div_topp_t5 = topp_t5["diversity_per_timestep"]
    div_topp_t10 = topp_t10["diversity_per_timestep"]

    div_topk_t0 = topk_t0["diversity_per_timestep"]
    div_topk_t1 = topk_t1["diversity_per_timestep"]
    div_topk_t2 = topk_t2["diversity_per_timestep"]
    div_topk_t5 = topk_t5["diversity_per_timestep"]
    div_topk_t10 = topk_t10["diversity_per_timestep"]

    div_multi_t0 = multi_t0["diversity_per_timestep"]
    div_multi_t1 = multi_t1["diversity_per_timestep"]
    div_multi_t2 = multi_t2["diversity_per_timestep"]
    div_multi_t5 = multi_t5["diversity_per_timestep"]
    div_multi_t10 = multi_t10["diversity_per_timestep"]

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set(style="whitegrid", font_scale=1.4)

    # Define color per sampling method
    color_map = {
        "Top-p": "#1f77b4",  # blue
        "Top-k": "#2ca02c",  # green
        "Multinomial": "#d62728",  # red
    }

    # Define line style per temperature
    linestyle_map = {
        "1": "solid",
        "2": "dashed",
        "5": "dashdot",
        "10": "dotted"
    }

    # Grouped line dictionary as you have it
    lines = {
        "Top-p ($T{=}1$)": div_topp_t1,
        "Top-p ($T{=}2$)": div_topp_t2,
        "Top-p ($T{=}5$)": div_topp_t5,
        "Top-p ($T{=}10$)": div_topp_t10,

        "Top-k ($T{=}1$)": div_topk_t1,
        "Top-k ($T{=}2$)": div_topk_t2,
        "Top-k ($T{=}5$)": div_topk_t5,
        "Top-k ($T{=}10$)": div_topk_t10,

        "Multinomial ($T{=}1$)": div_multi_t1,
        "Multinomial ($T{=}2$)": div_multi_t2,
        "Multinomial ($T{=}5$)": div_multi_t5,
        "Multinomial ($T{=}10$)": div_multi_t10,
    }

    # X-axis values (1-indexed timesteps)
    timesteps = np.arange(len(next(iter(lines.values())))) + 1

    # Plot
    plt.figure(figsize=(10, 6))
    for label, values in lines.items():
        method = label.split(" ")[0]  # e.g., 'Top-p'
        temperature = label.split("{=}")[-1].strip("$)")  # e.g., '1'

        color = color_map[method]
        linestyle = linestyle_map[temperature]

        plt.plot(timesteps, values, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Labels and title
    plt.xlabel("Timestep", fontsize=20)
    plt.ylabel("Spatial Standard Deviation [m]",fontsize=20)
    plt.title("Diversity per Timestep Across Sampling Strategies",fontsize=20)
    plt.legend(loc="upper left", ncol=2, fontsize="medium")
    plt.tight_layout()

    # Save for LaTeX
    plt.show()