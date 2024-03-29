from src.python.services.data_collection_service import DataCollectionService
from src.python.services.experiments_service import ExperimentsService
import argparse
from src.python.code_analyzer.code_analyzer import AnalysisLevel
from src.python.entities.entity import Language
from pathlib import Path


def dataset(args):
    DataCollectionService.create_dataset(args)


def tr_torrent(args):
    DataCollectionService.process_tr_torrent(args)


def learn(args):
    ExperimentsService.run_all_tsp_accuracy_experiments(args)


def decay_test(args):
    ExperimentsService.run_decay_test_experiments(args)


def results(args):
    ExperimentsService.statistical_analysis(args)

def train(args):
    args.output_path.mkdir(parents=True, exist_ok=True)
    ExperimentsService.run_experiments(args)


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--project-slug",
        help="The project's GitHub slug, e.g., apache/commons.",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--ci-data-path",
        help="Path to CI datasource root directory, including RTP-Torrent and Travis Torrent.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--test-path",
        help="Specifies the relative root directory of the test source code.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--level",
        help="Specifies the granularity of feature extraction.",
        type=AnalysisLevel,
        choices=[AnalysisLevel.FILE],
        default=AnalysisLevel.FILE,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    parser.add_argument(
        "--language",
        help="Project's main language",
        type=Language,
        choices=[Language.JAVA],
        default=Language.JAVA,
    )
    parser.add_argument(
        "-n",
        "--build-window",
        help="Specifies the number of recent builds to consider for computing features.",
        type=int,
        required=False,
        default=6,
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    tr_torrent_parser = subparsers.add_parser(
        "tr_torrent",
        help="Process travis torrent logs and build info.",
    )
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create training dataset including all test case features for each CI cycle.",
    )
    learn_parser = subparsers.add_parser(
        "learn",
        help="Perform learning experiments on collected features using RankLib.",
    )
    decay_test_parser = subparsers.add_parser(
        "decay_test",
        help="Perform ML ranking models decay test experiments on trained models.",
    )
    results_parser = subparsers.add_parser(
        "results",
        help="Analyze the results from experiments and generate tables.",
    )
    train_parser = subparsers.add_parser(
        "train",
        help="Analyze ",
    )

    add_dataset_parser_arguments(dataset_parser)
    dataset_parser.set_defaults(func=dataset)

    tr_torrent_parser.set_defaults(func=tr_torrent)
    tr_torrent_parser.add_argument(
        "-r",
        "--repo",
        help="The login and name of the repo seperated by @ (e.g., presto@prestodb)",
        type=str,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-i",
        "--input-path",
        help="Specifies the directory to of travis torrent raw data.",
        type=Path,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting data.",
        type=Path,
        default=".",
    )

    learn_parser.set_defaults(func=learn)
    learn_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    learn_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )

    decay_test_parser.set_defaults(func=decay_test)
    decay_test_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    decay_test_parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )

    results_parser.set_defaults(func=results)
    results_parser.add_argument(
        "-d",
        "--data-path",
        help="Path to the root folder of all datasets.",
        type=Path,
        default=None,
    )
    results_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting tables.",
        type=Path,
        default=".",
    )
    results_parser.add_argument(
        "-a",
        "--A",
        help="Specifies the spread sheet with the first measurement.",
        type=Path,
        default=".",
    )
    results_parser.add_argument(
        "-b",
        "--B",
        help="Specifies the spread sheet or directory with the second measurement(s).",
        type=Path,
        default=".",
    )

    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    train_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )
    train_parser.add_argument(
        "-r",
        "--ranker",
        help="Specifies the ranker for ranklib to use.",
        type=int,
        default=0,
    )
    train_parser.add_argument(
        "-p",
        "--params",
        help="Specifies the parameters for the ranker.",
        type=str,
        default='',
    )
    train_parser.add_argument(
        "-s",
        "--sampler",
        help="Specifies the parameters for the ranker.",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    args.unique_separator = "\t"
    args.func(args)


if __name__ == "__main__":
    main()
