import argparse

from actions import generate_data, operate
from interfaces import ArgsNamespace


def main():
    parser = argparse.ArgumentParser(description="Experiment utilities")
    subparsers = parser.add_subparsers(dest="command")

    __setup_data_generator(subparsers)
    __setup_operator(subparsers)

    args = parser.parse_args()

    __run_command(args)


def __run_command(args: ArgsNamespace):
    match args.command:
        case "generate-data":
            generate_data(args)

        case "operate":
            operate(args)


def __setup_data_generator(subparsers: argparse._SubParsersAction):
    data_generator_parser = subparsers.add_parser(
        "generate-data", help="Generate synthetic seismic data"
    )

    data_generator_parser.add_argument(
        "--inlines", type=int, default=100, help="Number of inlines"
    )
    data_generator_parser.add_argument(
        "--xlines", type=int, default=100, help="Number of crosslines"
    )
    data_generator_parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples"
    )
    data_generator_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for SEGY file"
    )
    data_generator_parser.add_argument(
        "--prefix", type=str, default="", help="Filename prefix"
    )


def __setup_operator(subparsers: argparse._SubParsersAction):
    operator_parser = subparsers.add_parser(
        "operate", help="Operate within a given operator"
    )

    operator_parser.add_argument(
        "operator",
        type=str,
        choices=["envelope"],
        help="Operator to use",
    )

    operator_parser.add_argument(
        "--segy-path", type=str, required=True, help="Path to SEGY file"
    )


if __name__ == "__main__":
    main()
