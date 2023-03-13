from parser.abstract_parser import AbstractArgumentParser


class TrainParser(AbstractArgumentParser):
    def add_common_arguments(self) -> None:
        self.parser.add_argument("--config", "-c", help="Path to the training configuration file.")
        self.parser.add_argument(
            "--train_data",
            "-d",
            help="Path to the training data file. (.npy file)",
            required=True,
        )
        self.parser.add_argument(
            "--result_dir",
            "-r",
            required=True,
            help="name of directory to store train artifacts. folder `../trained_models/{result_dir}` is created",
        )
        self.parser.add_argument(
            "--model_architecture",
            "-a",
            help="The name of the class that defines the model architecture.",
            choices=[
                "fully_connected_v1",
            ],
            default="fully_connected_v1",
        )
        self.parser.add_argument(
            "--epoch", type=int, default=100, help="Number of epochs to train for. Default is 100."
        )
        self.parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
        self.parser.add_argument(
            "--lr", type=float, default=3.3e-4, help="Learning rate for the optimizer. Default is 3.3e-4."
        )
        self.parser.add_argument(
            "--validation_fraction",
            "-v",
            type=float,
            default=0.1,
            help="Fraction of data to use for validation. Default is 0.1.",
        )
        self.parser.add_argument(
            "--cross_validation_split",
            "-k",
            help="number of cross validation splits. Default is 7(21 participants).",
            default=7,
            type=int,
        )
