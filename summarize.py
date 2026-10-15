"""Aggregate result CSVs into the numbers reported in the paper.

Accuracy is averaged within each model first and then across models, so
every model carries equal weight.

    python summarize.py --synthetic results/synthetic --remention results/remention --niah results/niah
"""

import argparse
from pathlib import Path

import pandas as pd


def load(root: str | None, exclude: list[str]) -> pd.DataFrame | None:
    if not root:
        return None
    files = sorted(Path(root).rglob("results.csv"))
    if not files:
        print(f"No results.csv under {root}")
        return None
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    df = df[~df["model"].isin(exclude)]
    if "task" not in df.columns:
        if "dataset_type" in df.columns:
            df["task"] = df["dataset_type"]
        else:
            df["task"] = (df["domain"] == "sentiment").map({True: "sentiment", False: "factual"})
    return df


def model_mean(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean accuracy / abstention per ``by`` group, averaged over models."""
    per_model = df.groupby(["model", *by])[["accuracy", "abstained"]].mean()
    return per_model.groupby(level=by).mean()


def show(title: str, table: pd.DataFrame | pd.Series) -> None:
    print(f"\n### {title}\n")
    print((table * 100).round(1).to_string())


def ordering(fmt: str) -> str:
    return "entity_first" if fmt.endswith("entity_first") else "attr_first"


def summarize_synthetic(df: pd.DataFrame) -> None:
    print(f"\n## Experiments 1-2: Synthetic Associations ({df['model'].nunique()} models)")
    order = ["no_eviction", "removed_from_context", "evict_both", "evict_both_keep_punct",
             "evict_attribute", "evict_entity"]
    for task, sub in df.groupby("task"):
        table = model_mean(sub, ["condition"]).reindex([c for c in order if c in set(sub["condition"])])
        show(f"{task}: accuracy / abstention (%) by condition", table)

    partial = df[df["condition"].isin(["evict_attribute", "evict_entity"])].copy()
    partial["ordering"] = partial["context_format"].map(ordering)
    for task, sub in partial.groupby("task"):
        table = model_mean(sub, ["question_direction", "ordering", "condition"])["accuracy"].unstack("condition")
        show(f"{task}: partial-eviction accuracy (%) by token order (Exp. 2)", table)


def remention_benefit(df: pd.DataFrame, kept: str, evicted: str, by: list[str]) -> pd.Series:
    acc = model_mean(df, by + ["condition"])["accuracy"].unstack("condition")
    return acc[kept] - acc[evicted]


def summarize_remention(df: pd.DataFrame) -> None:
    print(f"\n## Experiment 3: re-mention in controlled contexts ({df['model'].nunique()} models)")
    order = ["no_eviction", "evict_pair", "evict_pair_and_anchor"]
    for task, sub in df.groupby("task"):
        show(f"{task}: accuracy / abstention (%)", model_mean(sub, ["condition"]).reindex(order))
    benefit = remention_benefit(df, "evict_pair", "evict_pair_and_anchor", ["task"])
    show("Re-mention benefit (pp): evict_pair - evict_pair_and_anchor", benefit.rename("benefit_pp"))


def summarize_niah(df: pd.DataFrame) -> None:
    print(f"\n## Experiment 3: Needle-in-a-Haystack ({df['model'].nunique()} models: "
          f"{', '.join(sorted(df['model'].unique()))})")
    order = ["no_eviction", "evict_needle", "evict_needle_and_anchor", "evict_needle_no_anchor"]
    for task, sub in df.groupby("task"):
        show(f"{task}: accuracy / abstention (%), all cells", model_mean(sub, ["condition"]).reindex(order))
    for task, sub in df.groupby("task"):
        grid = remention_benefit(sub, "evict_needle", "evict_needle_and_anchor",
                                 ["needle_depth", "anchor_gap"]).unstack("anchor_gap")
        show(f"{task}: re-mention benefit (pp) by needle depth (rows) x gap (cols)", grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", default=None, help="Output root of run_synthetic.py")
    parser.add_argument("--remention", default=None, help="Output root of run_remention.py")
    parser.add_argument("--niah", default=None, help="Output root of run_niah.py")
    parser.add_argument("--exclude_models", nargs="*", default=[], help="Model ids to leave out")
    args = parser.parse_args()

    for root, fn in [(args.synthetic, summarize_synthetic), (args.remention, summarize_remention),
                     (args.niah, summarize_niah)]:
        df = load(root, args.exclude_models)
        if df is not None:
            fn(df)


if __name__ == "__main__":
    main()
