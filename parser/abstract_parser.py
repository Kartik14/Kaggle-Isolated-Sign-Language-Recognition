import argparse
from abc import ABC, abstractmethod

from helper.logging import logger
from helper.utils import read_yaml_file


class AbstractArgumentParser(ABC):
    """An abstract base class for argument parsers."""

    def __init__(self) -> None:
        # Create a parser object
        self.parser = argparse.ArgumentParser()

        # Add common arguments
        self.add_common_arguments()

    @abstractmethod
    def add_common_arguments(self) -> None:
        # Override this method to add common arguments for all subcommands
        pass

    def load_config_params(self, args: argparse.Namespace) -> argparse.Namespace:
        logger.info(f"Loading Arguments from config file {args.config}")
        config_params = read_yaml_file(args.config)
        self.parser.set_defaults(**config_params)
        return self.parser.parse_args()

    def parse_args(self) -> argparse.Namespace:
        # Parse the command-line arguments and return a namespace object
        args = self.parser.parse_args()

        if hasattr(args, "config") and args.config is not None:
            args = self.load_config_params(args)

        return args
