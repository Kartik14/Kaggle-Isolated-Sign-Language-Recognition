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
            "--input_conf",
            "-ic",
            help="path to input conf file",
            required=True,
        )
