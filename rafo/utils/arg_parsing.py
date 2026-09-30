from argparse import Namespace, ArgumentParser
from copy import deepcopy


def is_multiple_devices(devices: str) -> bool:
    if devices.isdigit():
        num_devices = int(devices)
        return num_devices > 1

    devices = devices.replace(" ", "")

    if "," in devices:
        if devices[-1] == ",":  # Remove trailing comma if present
            devices = devices[:-1]

        return len(devices.split(",")) > 1

    return False


def update_args_with_parser_default_vals(
    args: Namespace,
    *parsers: ArgumentParser
) -> Namespace:
    # I understand this isn't pretty but it gets the job done! :)
    args = deepcopy(args)
    args_dict = vars(args)

    for parser in parsers:
        defaults = {action.dest: action.default for action in parser._actions}

        for key in defaults:
            if key not in args_dict:
                args_dict[key] = defaults[key]

    return args
