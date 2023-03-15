from parser.abstract_parser import AbstractArgumentParser


class TrainParser(AbstractArgumentParser):
    def add_common_arguments(self) -> None:
        self.parser.add_argument("--config", "-c", help="Path to the training configuration file.")
        self.parser.add_argument(
            "--model_config",
            "-mc",
            help="config for model runtime_params",
            required=True,
        )
        self.parser.add_argument(
            "--result_dir",
            "-r",
            required=True,
            help="name of directory to store train artifacts. folder `../trained_models/{result_dir}` is created",
        )
        self.parser.add_argument(
            "--train_data",
            "-d",
            help="Path to the training data file. (.npy file)",
            required=True,
        )
        self.parser.add_argument(
            "--epoch", type=int, default=200, help="Number of epochs to train for. Default is 200."
        )
        self.parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
        self.parser.add_argument(
            "--lr", type=float, default=3.3e-4, help="Learning rate for the optimizer. Default is 3.3e-4."
        )
        self.parser.add_argument(
            "--cross_validation_splits",
            "-k",
            help="number of cross validation splits. Default is 7(21 participants).",
            default=4,
            type=int,
        )
        self.parser.add_argument(
            "--early_stopping_patience",
            "-es",
            type=int,
            help="patience for early stopping",
            default=25,
        )
        self.parser.add_argument(
            "--lr_reduce_mult",
            type=float,
            help="multiplier for reducing lr",
            default=0.5,
        )
        self.parser.add_argument(
            "-lr_reduce_patience",
            type=int,
            help="patience for reducing lr",
            default=10,
        )
