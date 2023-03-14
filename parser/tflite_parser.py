from parser.abstract_parser import AbstractArgumentParser


class TfLiteParser(AbstractArgumentParser):
    def __init__(self) -> None:
        super().__init__()

    def add_common_arguments(self) -> None:
        self.parser.add_argument(
            "--save_dir",
            "-s",
            help="Path to save the tflite model",
        )
        self.parser.add_argument(
            "--model_dirs",
            "-m",
            nargs="+",
            required=True,
            help="path(s) to trained model directory",
        )
        self.parser.add_argument(
            "--norm_stats",
            help="Path to normalisation stats",
        )
        self.parser.add_argument(
            "--input_mode",
            "-i",
            help="layer to use for processing input",
            default="frame_mean_std_v1",
        )
