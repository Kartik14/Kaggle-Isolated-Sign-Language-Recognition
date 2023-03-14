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
            default="frame_mean_std",
        )
        self.parser.add_argument("--skip_z", action="store_true", help="set to skip z coordinate")
        self.parser.add_argument("--expt", action="store_true", help="set to run on small part of data for expt")
        self.parser.add_argument(
            "--landmarks_set",
            "-l",
            nargs="+",
            default=["lips", "right_hand", "left_hand", "pose"],
            choices=list(constants.Landmarks.keys()),
            help="set of landmark features to use.",
        )
