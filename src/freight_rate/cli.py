"""Command-line entry point.

    python -m freight_rate.cli quality    # data-quality report
    python -m freight_rate.cli eda        # exploratory figures + summary -> reports/
    python -m freight_rate.cli validate   # holdout + rolling backtest -> reports/
    python -m freight_rate.cli train      # fit on all development data -> outputs/models/
    python -m freight_rate.cli predict    # write validation_predictions.csv + December inputs
    python -m freight_rate.cli score      # run Spotter's score.py on the outputs
    python -m freight_rate.cli all        # everything above, in order
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from . import config
from .data import load_all
from .eda import run_eda
from .features import FeatureBuilder
from .model import HybridConfig, HybridRateModel
from .predict import predict_december, predict_validation, write_outputs
from .validate import run_backtest, run_holdout, write_validation_report

MODEL_PATH = config.MODEL_DIR / "hybrid_rate_model.pkl"


def _model_factory(args) -> callable:
    cfg = HybridConfig(trend_mode=args.trend, n_seeds=args.seeds, num_boost_round=args.rounds, learning_rate=args.lr)
    return lambda: HybridRateModel(cfg)


def cmd_quality(bundle, args) -> None:
    for report in bundle["reports"]:
        print(report.to_markdown()); print()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "data_quality.md").write_text("\n\n".join(r.to_markdown() for r in bundle["reports"]), encoding="utf-8")
    print(f"wrote {config.REPORT_DIR / 'data_quality.md'}")


def cmd_eda(bundle, args) -> None:
    path = run_eda(bundle["train"], bundle["validation"])
    print(path.read_text(encoding="utf-8")); print(f"wrote {path}")


def cmd_validate(bundle, args) -> None:
    builder = FeatureBuilder(bundle["calendar"], bundle["cities"]); factory = _model_factory(args)
    t0 = time.time()
    holdout = run_holdout(bundle["train"], builder, factory)
    print(f"holdout done in {time.time()-t0:.0f}s"); print(holdout["comparison"].to_string(index=False)); print()
    backtest = run_backtest(bundle["train"], builder, factory) if not args.skip_backtest else pd.DataFrame()
    if len(backtest):
        print(backtest[["test_window", "MAE", "inlier_MAE", "inlier_MAPE", "inlier_bias_pct", "trend_pct_per_month"]].to_string(index=False))
    path = write_validation_report(holdout, backtest)
    print(f"\nwrote {path}")


def cmd_train(bundle, args) -> None:
    builder = FeatureBuilder(bundle["calendar"], bundle["cities"])
    train = bundle["train"]; fit = train[train["is_target_outlier"] == 0]
    t0 = time.time()
    model = _model_factory(args)().fit(builder.transform(fit), fit[config.TARGET], fit["distance"])
    model.save(MODEL_PATH)
    print(f"trained on {len(fit):,} loads ({train['is_target_outlier'].sum():,} corrupted targets excluded) in {time.time()-t0:.0f}s")
    print(json.dumps(model.summary(), indent=2))
    print(model.feature_importance().to_string(index=False))
    print(f"saved {MODEL_PATH}")


def cmd_predict(bundle, args) -> None:
    builder = FeatureBuilder(bundle["calendar"], bundle["cities"])
    model = HybridRateModel.load(MODEL_PATH)
    template = pd.read_csv(config.TEMPLATE_PATH)
    validation_predictions = predict_validation(model, builder, bundle["validation"], template)
    december_template = pd.read_csv(config.DECEMBER_INPUT_PATH)
    december = predict_december(model, builder, bundle["calendar"], december_template)
    paths = write_outputs(validation_predictions, december)
    print(validation_predictions["predicted_rate"].describe().round(2).to_string())
    print("\nDecember (Lexington -> Fort Wayne, Dry Van, 360 mi, 32,000 lb):")
    print(december[["date", "predicted_rate"]].to_string(index=False))
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


def cmd_score(bundle, args) -> None:
    cmd = [sys.executable, str(config.ROOT / "score.py"), "--predictions", str(config.ROOT / "validation_predictions.csv"),
           "--december-predictions", str(config.DECEMBER_INPUT_PATH), "--output-dir", str(config.ROOT / "scorer_results")]
    print(" ".join(cmd)); subprocess.run(cmd, check=True)


COMMANDS = {"quality": cmd_quality, "eda": cmd_eda, "validate": cmd_validate, "train": cmd_train, "predict": cmd_predict, "score": cmd_score}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Freight rate prediction pipeline")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--trend", default="full", choices=["full", "damped", "none"], help="how the fitted linear trend extends past the training window")
    parser.add_argument("--seeds", type=int, default=3, help="number of LightGBM boosters averaged")
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args(argv)

    bundle = load_all()
    steps = list(COMMANDS) if args.command == "all" else [args.command]
    for step in steps:
        print(f"\n===== {step} =====")
        COMMANDS[step](bundle, args)


if __name__ == "__main__":
    main()
