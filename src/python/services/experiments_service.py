import pandas as pd
from ..decay_dataset_factory import DecayDatasetFactory
from ..dataset_factory import DatasetFactory
from ..feature_extractor.feature import Feature
from ..module_factory import ModuleFactory
import sys
from ..ranklib_learner import RankLibLearner
from ..feature_extractor.feature import Feature
from ..code_analyzer.code_analyzer import AnalysisLevel
from ..results.results_analyzer import ResultAnalyzer
import os
from scipy.stats import wilcoxon
import pingouin as pg
import re

class ExperimentsService:
    @staticmethod
    def remove_outlier_tests(args, dataset_df):
        test_f = dataset_df[dataset_df[Feature.VERDICT] > 0][
            [Feature.TEST, Feature.BUILD]
        ]
        test_fcount = (
            test_f.groupby(Feature.TEST, as_index=False)
            .count()
            .sort_values(Feature.BUILD, ascending=False, ignore_index=True)
        )
        test_fcount["rate"] = (
            test_fcount[Feature.BUILD] / dataset_df[Feature.BUILD].nunique()
        )
        mean, std = test_fcount["rate"].mean(), test_fcount["rate"].std()
        outliers = []
        for _, r in test_fcount.iterrows():
            if abs(r["rate"] - mean) > 3 * std:
                outliers.append(int(r[Feature.TEST]))

        outliers_path = (
            args.output_path / "tsp_accuracy_results" / "full-outliers" / "outliers.csv"
        )
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"test": outliers}).to_csv(outliers_path, index=False)
        result_df = dataset_df[~dataset_df[Feature.TEST].isin(outliers)]
        failed_builds = (
            result_df[result_df[Feature.VERDICT] > 0][Feature.BUILD].unique().tolist()
        )
        return (
            result_df[result_df[Feature.BUILD].isin(failed_builds)]
            .copy()
            .reset_index(drop=True)
        )

    @staticmethod
    def run_all_tsp_accuracy_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        print(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            print(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        results_path = args.output_path / "tsp_accuracy_results"
        print("***** Running full feature set experiments *****")
        learner.run_accuracy_experiments(dataset_df, "full", results_path)
        learner.test_heuristics(dataset_df, results_path / "full")
        print("***** Running full feature set without Outliers experiments *****")
        outliers_dataset_df = ExperimentsService.remove_outlier_tests(args, dataset_df)
        learner.run_accuracy_experiments(
            outliers_dataset_df, "full-outliers", results_path
        )
        learner.test_heuristics(outliers_dataset_df, results_path / "full-outliers")
        print()
        print("***** Running w/o impacted feature set experiments *****")
        learner.run_accuracy_experiments(
            dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
            "wo-impacted",
            results_path,
        )
        print("***** Running w/o impacted feature set Outliers experiments *****")
        learner.run_accuracy_experiments(
            outliers_dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
            "wo-impacted-outliers",
            results_path,
        )
        print()
        feature_groups_names = {
            "TES_COM": Feature.TES_COM,
            "TES_PRO": Feature.TES_PRO,
            "TES_CHN": Feature.TES_CHN,
            "REC": Feature.REC,
            "COV": Feature.COV,
            "COD_COV_COM": Feature.COD_COV_COM,
            "COD_COV_PRO": Feature.COD_COV_PRO,
            "COD_COV_CHN": Feature.COD_COV_CHN,
            "DET_COV": Feature.DET_COV,
        }
        for feature_group, names in feature_groups_names.items():
            print(f"***** Running w/o {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                dataset_df.drop(names, axis=1), f"wo-{feature_group}", results_path
            )
            print(
                f"***** Running w/o {feature_group} feature set Outliers experiments *****"
            )
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(names, axis=1),
                f"wo-{feature_group}-outliers",
                results_path,
            )
            print()

        test_code_features = Feature.TES_COM + Feature.TES_PRO + Feature.TES_CHN
        test_execution_features = Feature.REC
        test_coverage_features = (
            Feature.COV
            + Feature.COD_COV_COM
            + Feature.COD_COV_PRO
            + Feature.COD_COV_CHN
            + Feature.DET_COV
        )
        high_level_feature_groups = {
            "Code": test_code_features,
            "Execution": test_execution_features,
            "Coverage": test_coverage_features,
        }
        non_feature_cols = [
            Feature.BUILD,
            Feature.TEST,
            Feature.VERDICT,
            Feature.DURATION,
        ]
        for feature_group, names in high_level_feature_groups.items():
            print(f"***** Running with {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                dataset_df[non_feature_cols + names], f"W-{feature_group}", results_path
            )
            print(f"***** Running with {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                outliers_dataset_df[non_feature_cols + names],
                f"W-{feature_group}-outliers",
                results_path,
            )
            print()

    @staticmethod
    def run_decay_test_experiments(args):
        print(f"Running decay tests for {args.output_path.name}")
        repo_miner_class = ModuleFactory.get_repository_miner(AnalysisLevel.FILE)
        repo_miner = repo_miner_class(args)
        change_history_df = repo_miner.load_entity_change_history()
        dataset_factory = DatasetFactory(
            args,
            change_history_df,
            repo_miner,
        )
        dataset_df = pd.read_csv(args.output_path / "dataset.csv")
        decay_ds_factory = DecayDatasetFactory(dataset_factory, args)
        models_path = args.output_path / "tsp_accuracy_results" / "full"
        decay_ds_factory.create_decay_datasets(dataset_df, models_path)

        learner = RankLibLearner(args)
        datasets_path = args.output_path / "decay_datasets"
        learner.run_decay_test_experiments(datasets_path, models_path)
        print(f"All finished and results are saved at {datasets_path}")
        print()

    @staticmethod
    def analyze_results(args):
        result_analyzer = ResultAnalyzer(args)
        result_analyzer.analyze_results()

    @staticmethod
    def collect_stats(results_dir):
        results_folders = os.listdir(results_dir)
        stats_df = pd.DataFrame()
        # gather apfdc info from each dataset
        all_results_df = pd.DataFrame()
        for results_folder in results_folders:
            folder = results_dir / results_folder
            if os.path.isdir(folder):
                # read apfdc values
                results_path = folder / "results.csv"
                results_df = pd.read_csv(results_path, usecols=['apfdc'])
                apfdc_stats_df = results_df['apfdc'].describe()
                # write apfdc info to dataframe
                row = {'dataset': results_folder, 'mean_apfdc': apfdc_stats_df['mean'], 'std_apfdc': apfdc_stats_df['std'], '50%_apfdc': apfdc_stats_df['50%']}
                stats_df = stats_df.append(row, ignore_index=True)
                # accumulate apfdc values for all datasets
                all_results_df = all_results_df.append(results_df, ignore_index=True)
        # write stats for all datasets into same folder
        all_apfdc_stats_df = all_results_df['apfdc'].describe()
        row = {'dataset': 'ALL', 'mean_apfdc': all_apfdc_stats_df['mean'], 'std_apfdc': all_apfdc_stats_df['std'], '50%_apfdc': all_apfdc_stats_df['50%']}
        stats_df = stats_df.append(row, ignore_index=True)
        stats_df = stats_df[['dataset', 'mean_apfdc', 'std_apfdc', '50%_apfdc']]
        stats_df.to_csv(results_dir / "apfdc_stats.csv", index=False)

    @staticmethod
    def run_experiments(args):
        i = 0
        while i < 25:
            i = i+1
            dataset_path = args.output_path / f"dataset{i}.csv"
            if not dataset_path.exists():
                print(f"No dataset{i}.csv found in the output directory.")
                continue
            print(f"##### Running experiments for dataset{i}.csv #####")
            learner = RankLibLearner(args)
            dataset_df = pd.read_csv(dataset_path)
            builds_for_training = 10
            builds_count = dataset_df[Feature.BUILD].nunique()
            if builds_count <= builds_for_training:
                print(f"Not enough builds for training: require at least {builds_for_training + 1}, found {builds_count}")
                continue
            results_path = args.output_path / "results" / f"results{i}"
            builds_path = args.output_path / f"builds{i}.csv"
            learner.run_experiments(dataset_df, results_path, builds_path)
        # collect apfdc stats and write results
        ExperimentsService.collect_stats(args.output_path / "results")

    @staticmethod
    def statistical_analysis(args):
        a_path, b_path, out_path = args.A, args.B, args.output_path

        # validate paths
        if len({a_path, b_path, out_path}) != 3:
            print('Expected 3 arguments with distinct values. Aborting.')
            return
        if not a_path.exists():
            print(f'No file {a_path} found. Aborting.')
            return
        if not b_path.exists():
            print(f'No file or directory {b_path} found. Aborting.')
            return
        if os.path.isfile(out_path):
            print(f'Output file already exists. Aborting.')
            return

        # first file in comparison
        a_filename = os.path.basename(a_path)
        # omit last row since it's the average
        a_df = pd.read_csv(a_path)[:-1]
        a = a_df['mean_apfdc'].values

        # second file(s) in comparison
        if os.path.isfile(b_path) and b_path is not a_path:
            files = [b_path]
        else:
            # fullpath of all csv in directory that are not first file in comparison
            files = [b_path / f for f in os.listdir(b_path) if re.match(r'.*\.csv', f) and f != a_filename]
        if len(files) == 0:
            print('No files to compare. Aborting.')
            return
        b_list = []
        for file in files:
            # omit last row since it's the average
            b_df = pd.read_csv(file)[:-1]
            b = b_df['mean_apfdc'].values
            b_filename = os.path.basename(file)
            b_name = b_filename[0:b_filename.rfind('.')]
            b_list.append([b_name, b])

        # do comparison
        result = {"method": [], "p-value": [], "CLES": []}
        for b in b_list:
            z, p = wilcoxon(b[1], a)
            cles = pg.compute_effsize(b[1], a, paired=True, eftype="CLES")
            result["method"].append(b[0])
            result["p-value"].append(p)
            result["CLES"].append(cles)
        result_df = pd.DataFrame(result).sort_values("p-value", ignore_index=True)
        result_df.to_csv(out_path, index=False)