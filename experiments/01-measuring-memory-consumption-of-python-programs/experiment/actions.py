from interfaces import ArgsNamespace


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
    from data import load_segy

    match args.operator:
        case "envelope":
            from operators import envelope

            data = load_segy(args.segy_path)
            envelope.envelope_from_ndarray(data)
