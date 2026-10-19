from mrock.get_data import DataLoader
from pathlib import Path
import math

def find_available_temperatures(parameters, file_pattern):
    loader = DataLoader()
    base_directory = loader._to_path(
        "nickel_cut",
        parameters["subdir"],
        L=parameters["L"],
    )
    relative_base = base_directory.relative_to(loader.data_dir)
    all_files = loader.get_all_files(relative_base, file_pattern)

    fixed_parameters = {
        key: parameters[key] for key in ("L", "U_0", "tprime", "E_F")
    }
    available = set()

    for file in all_files:
        path = Path(file)

        components = {}
        for part in path.parent.parts:
            if "=" in part:
                key, value = part.split("=", 1)
                components[key] = value

        try:
            matches_fixed_parameters = all(
                math.isclose(
                    float(components[key]),
                    float(value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for key, value in fixed_parameters.items()
            )
            temperature = float(components["T"])
        except (KeyError, ValueError):
            continue

        if matches_fixed_parameters:
            available.add(temperature)

    return sorted(available)