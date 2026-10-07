import argparse
import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import AMAZONBEAUTY, AMAZONCD, AMAZONTOYSANDGAMES, MOVIELENS, YELP, GRAPHS_DIR
from knowlight.graphs.tag_builder import TAGGraphBuilder

SUPPORTED_DATASETS = [AMAZONBEAUTY, AMAZONCD, AMAZONTOYSANDGAMES, MOVIELENS, YELP]


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyG Text-Attributed Graph (TAG) datasets with raw textual metadata."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=SUPPORTED_DATASETS + ["all"],
        help="Dataset name to process ('amazon_beauty', 'amazon_cd_2014', 'amazon_toys_and_games', 'movielens', 'yelp', or 'all')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(GRAPHS_DIR),
        help="Directory where generated PyG graph files (.pt) will be saved"
    )

    args = parser.parse_args()

    target_datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("  PyG Text-Attributed Graph (TAG) Generator (Raw Text Attributes)")
    print(f"  Target Datasets: {target_datasets}")
    print(f"  Output Directory: {output_dir.resolve()}")
    print("=" * 70)

    for ds in target_datasets:
        print(f"\n Processing dataset: '{ds}'...")
        try:
            builder = TAGGraphBuilder(dataset_name=ds)
            graph = builder.build_graph()

            # Save graph
            out_path = output_dir / f"{ds}_tag_graph.pt"
            builder.save_graph(graph, output_path=out_path)

            print(f" Dataset '{ds}' successfully processed -> {out_path}")
        except FileNotFoundError as e:
            print(f" [Skipped] Dataset '{ds}' is missing interaction TSV files: {e}")
        except Exception as e:
            print(f" [Error] Failed to generate TAG graph for '{ds}': {e}")

    print("\n" + "=" * 70)
    print(" Finished processing all requested datasets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
