from os.path import join
from parser.abstract_parser import AbstractArgumentParser

import constants


class PreprocessParser(AbstractArgumentParser):
    def add_common_arguments(self) -> None:
        self.parser.add_argument(
            "--config",
            "-c",
            help="python to config file",
        )
        self.parser.add_argument(
            "--train_csv",
            default=join(constants.DATA_ROOT, "train.csv"),
            help="path to train csv file",
        )
        self.parser.add_argument(
            "--output_path",
            "-o",
            help="path to save processed data",
            required=True,
        )
        self.parser.add_argument(
            "--mode",
            "-m",
            help="mode to use for preprocessing",
            default="frame_mean_std_v1",
        )
        self.parser.add_argument("--skip_z", action="store_true", help="set to skip z coordinate")
        self.parser.add_argument("--expt_run", action="store_true", help="set to run on small part of data for expt")
        self.parser.add_argument(
            "--feature_set",
            nargs="+",
            default="all",
            choices=["all", "face", "left_hand", "right_hand", "pose"],
            help="Subset of landmark features to use. Default is all features.",
        )
