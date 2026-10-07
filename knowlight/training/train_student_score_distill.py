import argparse
import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.training.train_student import main as train_score_main


def main():
    parser = argparse.ArgumentParser(description="Pipeline 1: KnowLight Score Distillation (GKD/LinguGKD Inspired)")
    parser.add_argument("--config", type=str, default="config/train_student_score_distill.yaml", help="Path to score distillation train config YAML")
    parser.add_argument("--llm_config", type=str, default="config/teacher_llm.yaml", help="Path to LLM teacher config file")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Name of the LLM teacher")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--grid", action="store_true", default=False, help="Enable Distillation Grid Search")
    parser.add_argument("--grid_search", action="store_true", default=False, help="Enable LR/WD Grid Search")
    parser.add_argument("--beta", nargs="+", type=float, default=None, help="Beta values to test in score grid search")
    parser.add_argument("--distiller", type=str, default="score", choices=["score", "knowlight", "rrd", "unkd"], help="Distillation method to use: 'score' (default), 'rrd', 'unkd'")
    parser.add_argument("--lambda_rrd", nargs="+", type=float, default=None, help="Distillation weight(s) for RRD (grid default: [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0])")
    parser.add_argument("--interesting_size", type=int, default=None, help="Interesting items per user for RRD (default: 40)")
    parser.add_argument("--uninteresting_size", type=int, default=None, help="Uninteresting items per user for RRD (default: 50)")
    parser.add_argument("--temperature_rrd", type=float, default=None, help="Sampling temperature for RRD (default: 1.0)")
    parser.add_argument("--lambda_unkd", nargs="+", type=float, default=None, help="Distillation weight(s) for UnKD (grid default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])")
    parser.add_argument("--sample_num", type=int, default=None, help="Sample number for UnKD (default: 30)")
    parser.add_argument("--group_count", type=int, default=None, help="Popularity group count for UnKD (default: 4)")
    parser.add_argument("--no_distillation", action="store_true", default=False, help="Disable distillation for pure LightGCN baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Pass arguments to main runner
    sys.argv = [sys.argv[0]] + [
        "--config", args.config,
        "--llm_config", args.llm_config,
        "--teacher", args.teacher,
        "--dataset", args.dataset,
        "--distiller", args.distiller,
        "--seed", str(args.seed)
    ]
    if args.epochs:
        sys.argv.extend(["--epochs", str(args.epochs)])
    if args.grid:
        sys.argv.append("--grid")
    if args.grid_search:
        sys.argv.append("--grid_search")
    if args.beta:
        sys.argv.extend(["--beta"] + [str(b) for b in args.beta])
    if getattr(args, "no_distillation", False):
        sys.argv.append("--no_distillation")
    if args.lambda_rrd is not None:
        sys.argv.extend(["--lambda_rrd"] + [str(l) for l in args.lambda_rrd])
    if args.interesting_size is not None:
        sys.argv.extend(["--interesting_size", str(args.interesting_size)])
    if args.uninteresting_size is not None:
        sys.argv.extend(["--uninteresting_size", str(args.uninteresting_size)])
    if args.temperature_rrd is not None:
        sys.argv.extend(["--temperature_rrd", str(args.temperature_rrd)])
    if args.lambda_unkd is not None:
        sys.argv.extend(["--lambda_unkd"] + [str(l) for l in args.lambda_unkd])
    if args.sample_num is not None:
        sys.argv.extend(["--sample_num", str(args.sample_num)])
    if args.group_count is not None:
        sys.argv.extend(["--group_count", str(args.group_count)])

    train_score_main()


if __name__ == "__main__":
    main()
