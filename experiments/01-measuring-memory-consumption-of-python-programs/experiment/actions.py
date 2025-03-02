from interfaces import ArgsNamespace
from profilers import profile_memory_usage

__all__ = ["generate_data", "operate"]


def generate_data(args: ArgsNamespace):
    from data import generate_seismic_data

    generate_seismic_data(
        inlines=args.inlines,
        xlines=args.xlines,
        samples=args.samples,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


def operate(args: ArgsNamespace):
    match args.operator:
        case "envelope":
            __run_envelope(args)
        case _:
            raise ValueError(f"Operator '{args.operator}' is not supported.")


def __run_envelope(args: ArgsNamespace):
    from data import load_segy
    from operators import envelope

    data = load_segy(args.segy_path)
    (
        profile_memory_usage(envelope.envelope_from_ndarray, args, data)
        if args.memory_profiler
        else envelope.envelope_from_ndarray(data)
    )
