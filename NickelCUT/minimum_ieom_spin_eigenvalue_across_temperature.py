import matplotlib.pyplot as plt
import numpy as np

from ieom import (
    compute_dynamical_matrix_spin,
    compute_occupation_numbers,
    hartree_fock,
)
from load_flow_files import find_available_temperatures, load_final_flow_state
from Momentum import Momentum
from nickel_cut_parameters import FLOW_PARAMETERS


FINAL_STATE_PATTERN = "final_flow_state.json.gz*"
FINAL_STATE_RE = r"^final_flow_state\.json\.gz\d*$"
NEAR_ZERO_TOLERANCE = 1e-8


def main():
    temperatures = find_available_temperatures(
        FLOW_PARAMETERS,
        FINAL_STATE_PATTERN,
        FINAL_STATE_RE,
    )
    if not temperatures:
        raise RuntimeError("No final flow states found for the selected parameters.")

    plot_temperatures = []
    minimum_eigenvalues = []
    for temperature in temperatures:
        parameters = {**FLOW_PARAMETERS, "T": temperature}
        flow_data = load_final_flow_state(
            **parameters,
            resume_num="",
            force_json=False,
        )
        beta = 1.0 / temperature if temperature > 0.0 else -1.0
        try:
            dispersion = hartree_fock(flow_data, beta)
        except RuntimeError as error:
            print(
                f"Skipping flow state T={temperature:g}: "
                f"Hartree-Fock did not converge ({error})"
            )
            continue

        occupations = compute_occupation_numbers(beta, dispersion)
        transfer_momentum = Momentum(flow_data["L"], 0, 0)
        dynamical_matrix = compute_dynamical_matrix_spin(
            flow_data,
            occupations,
            transfer_momentum,
        )
        eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
        non_near_zero_eigenvalues = eigenvalues[
            np.abs(eigenvalues) > NEAR_ZERO_TOLERANCE
        ]
        if non_near_zero_eigenvalues.size == 0:
            print(
                f"Skipping flow state T={temperature:g}: all eigenvalues are near zero "
                f"(|lambda| <= {NEAR_ZERO_TOLERANCE:g})"
            )
            continue
        minimum_eigenvalue = non_near_zero_eigenvalues[0]
        plot_temperatures.append(temperature)
        minimum_eigenvalues.append(minimum_eigenvalue)

    _, ax = plt.subplots(layout="constrained")
    ax.plot(plot_temperatures, minimum_eigenvalues, marker="o")
    ax.set_xlabel(r"$T=1/\beta$")
    ax.set_ylabel(
        r"$\min_{|\lambda|>10^{-8}} \lambda(M_{\mathrm{spin}}((-\pi,-\pi)))$"
    )
    ax.set_title("Minimum nonzero spin IEOM eigenvalue versus temperature")
    ax.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
