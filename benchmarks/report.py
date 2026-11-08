"""Print the README and EVALUATION_REPORT tables from ``benchmarks/results``.

uv run python benchmarks/report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adversarial_robustness_lab.stats import binom_test

RESULTS = Path(__file__).resolve().parent / "results"
ZOO = [
    ("small_cnn_standard", "Small CNN", "standard"),
    ("wrn16_2_standard", "WRN-16-2", "standard"),
    ("wrn16_2_fgsm_rs", "WRN-16-2", "FGSM-RS (fast AT)"),
    ("wrn16_2_pgd", "WRN-16-2", "PGD-AT (5 steps)"),
    ("wrn16_2_trades", "WRN-16-2", "TRADES (beta 6)"),
]
COHEN = {
    0.25: {"0.00": 0.77, "0.25": 0.61, "0.50": 0.42, "0.75": 0.25, "1.00": 0.00},
    0.50: {"0.00": 0.66, "0.25": 0.55, "0.50": 0.43, "0.75": 0.32, "1.00": 0.22},
}


def load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min"


def attack_acc(res: dict[str, Any], key: str) -> float | None:
    for row in res["attacks"]:
        if row["attack"] == key:
            return float(row["robust_accuracy"])
    return None


def headline() -> None:
    print(
        "| Model | Training | Parameters | Epochs | Training time | Clean | FGSM | PGD-20 | "
        "Ensemble (95% CI) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for name, arch, method in ZOO:
        res = load(RESULTS / "robustness" / f"{name}.json")
        if res is None:
            continue
        ens = res.get("ensemble")
        ens_txt = "n/a"
        if ens:
            lo, hi = ens["ci95"]
            ens_txt = f"**{pct(ens['robust_accuracy'])}** ({pct(lo)} to {pct(hi)})"
        print(
            f"| {arch} | {method} | {res['parameters'] / 1e6:.2f} M | {res['epochs']} | "
            f"{minutes(res['train_seconds'])} | {pct(res['clean_accuracy'])} | "
            f"{pct(attack_acc(res, 'fgsm'))} | {pct(attack_acc(res, 'pgd20'))} | {ens_txt} |"
        )


def ensemble_breakdown() -> None:
    print(
        "| Model | Clean | After APGD-CE | After APGD-T | After Square | Time (APGD-CE / APGD-T / Square) |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for name, arch, method in ZOO:
        res = load(RESULTS / "robustness" / f"{name}.json")
        if res is None or not res.get("ensemble"):
            continue
        ens, n = res["ensemble"], res["n"]
        after, sec = ens["robust_after"], ens["stage_seconds"]
        print(
            f"| {arch} {method} | {after['clean'] / n:.2%} | {after['apgd-ce'] / n:.2%} | "
            f"{after['apgd-t'] / n:.2%} | {after['square'] / n:.2%} | "
            f"{sec['apgd-ce']:.0f} s / {sec['apgd-t']:.0f} s / {sec['square']:.0f} s |"
        )


def cw_margin() -> None:
    print("| Model | PGD-20 (CE) | PGD-20 (C&W margin) |")
    print("|---|---:|---:|")
    for name, arch, method in ZOO:
        res = load(RESULTS / "robustness" / f"{name}.json")
        if res is None:
            continue
        print(
            f"| {arch} {method} | {pct(attack_acc(res, 'pgd20'))} | {pct(attack_acc(res, 'cw_pgd20'))} |"
        )


def verdict(row: dict[str, Any]) -> str:
    a, b = row["only_ours_breaks"], row["only_torchattacks_breaks"]
    if a + b == 0:
        return "tie (identical outcomes)"
    p = binom_test(a, a + b, 0.5)
    if p >= 0.05:
        return f"tie (p = {p:.2f})"
    return "ours stronger" if a > b else "torchattacks stronger"


def comparison() -> None:
    for name in ("wrn16_2_pgd", "wrn16_2_standard"):
        res = load(RESULTS / "compare" / f"{name}.json")
        if res is None:
            continue
        print(f"\n{name} (torchattacks {res['torchattacks']}):\n")
        print(
            "| Attack | Images | Ours: robust acc | torchattacks: robust acc | Broken only by ours / only by torchattacks | Verdict | Time ours / torchattacks |"
        )
        print("|---|---:|---:|---:|---:|---|---:|")
        for row in res["rows"]:
            o, t = row["ours"], row["torchattacks"]
            print(
                f"| {row['attack']} | {row['n']:,} | {pct(o['robust_accuracy'], 2)} | "
                f"{pct(t['robust_accuracy'], 2)} | {row['only_ours_breaks']} / "
                f"{row['only_torchattacks_breaks']} | {verdict(row)} | "
                f"{o['seconds']:.0f} s / {t['seconds']:.0f} s |"
            )


def certification() -> None:
    radii = ["0.00", "0.25", "0.50", "0.75", "1.00"]
    print(
        "| Base model | sigma | Images | n | "
        + " | ".join(f"r = {r}" for r in radii)
        + " | Abstain |"
    )
    print("|---|---:|---:|---:|" + "---:|" * len(radii) + "---:|")
    for name in ("wrn16_2_noise025", "wrn16_2_noise050"):
        res = load(RESULTS / "certify" / f"{name}.json")
        if res is None:
            continue
        top = max(res["budgets"], key=int)
        b = res["budgets"][top]
        cells = " | ".join(pct(b["certified_accuracy"][r]) for r in radii)
        print(
            f"| WRN-16-2, noise {res['train_sigma']} | {res['sigma']} | {res['images']} | {int(top):,} | {cells} | {pct(b['abstain_rate'])} |"
        )
    for sigma, row in COHEN.items():
        cells = " | ".join(pct(row[r], 0) for r in radii)
        print(f"| Cohen et al. ResNet-110 (cited) | {sigma} | 10,000 | 100,000 | {cells} | n/a |")


def budget_scaling() -> None:
    print(
        "| sigma | n | Certified at r = 0 | r = 0.25 | r = 0.5 | Abstain | Largest certifiable radius |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("wrn16_2_noise025", "wrn16_2_noise050"):
        res = load(RESULTS / "certify" / f"{name}.json")
        if res is None:
            continue
        for n, b in sorted(res["budgets"].items(), key=lambda kv: int(kv[0])):
            c = b["certified_accuracy"]
            print(
                f"| {res['sigma']} | {int(n):,} | {pct(c['0.00'])} | {pct(c['0.25'])} | {pct(c['0.50'])} | "
                f"{pct(b['abstain_rate'])} | {b['max_certifiable_radius']:.2f} |"
            )


def sanity() -> None:
    res = load(RESULTS / "sanity.json")
    if res is None:
        return
    models = list(res["results"])
    names = [c["name"] for c in res["results"][models[0]]["checks"]]
    print("| Check | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for i, check in enumerate(names):
        cells = []
        for m in models:
            checks = res["results"][m]["checks"]
            c = checks[i] if i < len(checks) else None
            cells.append(
                "n/a" if c is None else ("pass" if c["passed"] else "**FAIL**") + f": {c['detail']}"
            )
        print(f"| {check} | " + " | ".join(cells) + " |")
    for m in models:
        extra = res["results"][m]
        if "pgd20_bpda" in extra:
            print(
                f"\nQuantized model ({m}): clean {pct(extra['clean'])}, PGD-20 {pct(extra['pgd20'])}, "
                f"PGD-20 with BPDA {pct(extra['pgd20_bpda'])}, Square-1000 {pct(extra['square1000'])}."
            )


def ablations() -> None:
    res = load(RESULTS / "ablations.json")
    if res is None:
        return
    print(f"\nPGD steps ({res['model']}, first {res['n']} test images):\n")
    steps = res["pgd_steps"]
    print("| Steps | " + " | ".join(steps) + " |")
    print("|---|" + "---:|" * len(steps))
    print("| Robust accuracy | " + " | ".join(pct(v) for v in steps.values()) + " |")
    print(
        "\nRestarts (PGD-20): "
        + ", ".join(f"{k}: {pct(v)}" for k, v in res["pgd_restarts"].items())
    )
    print("\n| Iterations | PGD | APGD-CE |")
    print("|---:|---:|---:|")
    for it, v in res["pgd_vs_apgd"].items():
        print(f"| {it} | {pct(v['pgd'])} | {pct(v['apgd_ce'])} |")
    sq = res["square_queries"]
    print("\nSquare queries: " + ", ".join(f"{k}: {pct(v)}" for k, v in sq.items()))
    print(f"median queries to success: {res['square_median_queries_fooled']}")
    curves = res["eps_curve"]
    if curves:
        grid = list(next(iter(curves.values())))
        print("\n| Model | " + " | ".join(f"{e}/255" for e in grid) + " |")
        print("|---|" + "---:|" * len(grid))
        for m, c in curves.items():
            print(f"| {m} | " + " | ".join(pct(v) for v in c.values()) + " |")


def main() -> None:
    for title, fn in [
        ("Headline", headline),
        ("Ensemble breakdown", ensemble_breakdown),
        ("C&W margin loss", cw_margin),
        ("Comparison with torchattacks", comparison),
        ("Certification", certification),
        ("Sampling budget", budget_scaling),
        ("Sanity checks", sanity),
        ("Ablations", ablations),
    ]:
        print(f"\n### {title}\n")
        fn()


if __name__ == "__main__":
    main()
