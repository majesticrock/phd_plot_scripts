import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import string

import mrock.continued_fraction as cf

from .spectral_peak_analyzer import analyze_peak
from .legend import  *
from .continuum_all_data_pickler import DATA_CUTS
from .create_figure import *
from .make_panels_touch import make_panels_touch

CBAR_MAX = 10
CBAR_EXP = 3

# Settings the importer
BUILD_DIR = "plots/build/"
G_MAX_LOAD = 4.2
G_MAX_PLOT = 3.4

MEV = 1e3
INV_MEV = 1e-3

BEGIN_OFFSET = 1e1
RANGE = 1e1
SECOND_BEGIN = 1e-5
SECOND_RANGE = 1e-5
PEAK_PROMINCE = 1.
FIT_COMPLEX_SHIFT = 1e-8j

CONTINUUM_CUT_SHIFT = 1e-2

SIGMA = 0.00025 #eV

def gaussian_bell(x, mu):
    return np.exp(-0.5 * ((x - mu) / SIGMA)**2) / (SIGMA * np.sqrt(2 * np.pi))

def derivative_gaussian_bell(x, mu):
    return -((x - mu) / SIGMA**2) * gaussian_bell(x, mu)

def is_phase_peak(peak, coulomb):
    if coulomb:
        return False
    return abs(peak) < 1.3e-1 * INV_MEV

def extract_model_settings(ds):
    return f"g={ds['g']}  coulomb={ds['coulomb_scaling']}  screening={ds['lambda_screening']}"

class HeatmapPlotter:
    def __init__(self, data_frame_param, parameter_name, xlabel, zlabel=r'$A$ [$\mathrm{eV}^{-1}$]', xscale="linear", yscale="linear",
                 energy_range=(1e-10, 55.), scale_energy_by_gaps=False):
        self.data_frame = data_frame_param.sort_values(parameter_name).reset_index(drop=True)
        
        self.y = np.linspace(energy_range[0], energy_range[1], 10000) # meV
        self.x = (self.data_frame[parameter_name]).to_numpy()
        self.scale_energy_by_gaps = scale_energy_by_gaps
        self.resolvents = [cf.ContinuedFraction(pd_row, messages=False, ignore_first=80, ignore_last=88) for index, pd_row in self.data_frame.iterrows()]
        self.max_gaps   = np.array([2 * gap for gap in self.data_frame["Delta_max"]]) # meV
        self.true_gaps  = np.array([MEV * float(t_gap[0]) for t_gap in self.data_frame["continuum_boundaries"]]) # meV
        self.N_data = len(self.max_gaps)
        
        self.g_cuts = np.zeros(len(DATA_CUTS))
        for i in range(len(DATA_CUTS)):
            filtered_df = self.data_frame[self.data_frame['Delta_max'] < DATA_CUTS[i]]
            if len(filtered_df) == 0:
                self.g_cuts[i] = 0
            else:
                closest_row = filtered_df.loc[(DATA_CUTS[i] - filtered_df['Delta_max']).idxmin()]
                self.g_cuts[i] = closest_row['g']

        self.xlabel = xlabel
        self.zlabel = zlabel
        self.xscale = xscale
        self.yscale = yscale

    def __to_eV__(self, x, i):
        if self.scale_energy_by_gaps:
            return INV_MEV * self.max_gaps[i] * x
        return INV_MEV * x

    def uses_coulomb(self, i):
        return self.data_frame["coulomb_scaling"].iloc[i] != 0

    def __get_error__(self, key, i):
        __std_g__ = 0.05
        base = self.data_frame["g"].iloc[i]
        
        filter_lower = self.data_frame["g"][(self.data_frame["g"] < base - __std_g__)]
        filter_upper = self.data_frame["g"][(self.data_frame["g"] > base + __std_g__)]
        
        lower_err_idx = filter_lower.idxmax() if len(filter_lower) > 0 else -1
        upper_err_idx = filter_upper.idxmin() if len(filter_upper) > 0 else 2**31 - 1
        
        if key == "true_gap":
            return [ np.abs(self.true_gaps[i] - self.true_gaps[lower_err_idx]) if lower_err_idx >= 0                  else None,
                     np.abs(self.true_gaps[i] - self.true_gaps[upper_err_idx]) if upper_err_idx < len(self.true_gaps) else None ]
            
        lower_err = np.abs(self.data_frame[key].iloc[i] - self.data_frame[key].iloc[lower_err_idx]) if lower_err_idx >= 0                  else None
        upper_err = np.abs(self.data_frame[key].iloc[i] - self.data_frame[key].iloc[upper_err_idx]) if upper_err_idx < len(self.true_gaps) else None
        return [lower_err, upper_err]

    def fit_goldstone_peak(self, _real, _imag, i):
        """
        Fit the Goldstone / phase peak at omega = 0.

        Energies passed to the continued fraction are in eV.
        The plotting grid self.y and gaps are in meV, therefore we keep using
        scaling=INV_MEV as in the old version of this class.
        """
        __result = analyze_peak(
            _real,
            _imag,
            peak_position         = 0,
            range                 = RANGE,
            begin_offset          = BEGIN_OFFSET,
            scaling               = INV_MEV,
            reversed              = False,
            lower_continuum_edge  = INV_MEV * self.true_gaps[i],
            peak_pos_range        = self.y[20] - self.y[0],
            improve_peak_position = False
        )

        current_range = RANGE
        current_offset = BEGIN_OFFSET

        __best_fit = __result

        def deviation(_current):
            return abs(_current.slope.nominal_value + 2)

        def break_condition():
            return abs(__result.slope.nominal_value + 2) > 0.2

        while break_condition() and current_range >= SECOND_RANGE:
            current_offset = BEGIN_OFFSET

            while break_condition() and current_offset >= SECOND_BEGIN:
                __result = analyze_peak(
                    _real,
                    _imag,
                    peak_position         = __result.position,
                    range                 = current_range,
                    begin_offset          = __result.position + current_offset,
                    scaling               = INV_MEV,
                    reversed              = False,
                    lower_continuum_edge  = INV_MEV * self.true_gaps[i],
                    improve_peak_position = False
                )

                if deviation(__result) < deviation(__best_fit):
                    __best_fit = __result
                    best_range = current_range
                    best_offset = current_offset

                current_offset *= 0.5

            current_range *= 0.5

        __result = __best_fit

        if abs(__result.slope.nominal_value + 2) > 0.33:
            print("WARNING in Phase! Expected slope of '-2' does not match fitted slope!")
            print(__result)
            print(extract_model_settings(self.data_frame.iloc[i]), "\n")

        return __result

    def compute_higgs_peaks(self):
        """
        Compute Higgs bound-state positions and weights using the C++/continued-fraction
        bound-state classifier, as in the reference HeatmapPlotter.

        Returned peak positions are in eV.
        """
        higgs_cpp_results = [
            resolvent.classify_bound_states(
                "amplitude_SC",
                weight_domega=1e-8
            )
            for resolvent in self.resolvents
        ]

        __higgs_peak_positions = [
            np.array([data[0] for data in cpp_result])
            for cpp_result in higgs_cpp_results
        ]

        __higgs_peak_weights = [
            np.array([data[1] for data in cpp_result])
            for cpp_result in higgs_cpp_results
        ]

        return __higgs_peak_positions, __higgs_peak_weights

    def compute_phase_peaks(self):
        """
        Compute phase bound-state positions and weights using the same logic as
        the reference HeatmapPlotter.

        Non-Goldstone phase modes are obtained from classify_bound_states.
        The Goldstone mode at omega = 0 is inserted and fitted separately,
        unless Coulomb is active.

        Returned peak positions are in eV.
        """
        phase_cpp_results = []

        for i, resolvent in enumerate(self.resolvents):
            phase_cpp_results.append(
                resolvent.classify_bound_states(
                    "phase_SC",
                    weight_domega=1e-8,
                    is_phase_peak=lambda omega, i=i: is_phase_peak(
                        omega,
                        self.uses_coulomb(i)
                    )
                )
            )

        __phase_peak_positions = [
            [data[0] for data in cpp_result]
            for cpp_result in phase_cpp_results
        ]

        __phase_peak_weights = [
            [data[1] for data in cpp_result]
            for cpp_result in phase_cpp_results
        ]

        for i, res in enumerate(self.resolvents):
            # With Coulomb interaction, the Goldstone mode is lifted;
            # do not insert an omega=0 derivative-delta peak.
            if self.uses_coulomb(i):
                continue

            if self.max_gaps[i] < RANGE + BEGIN_OFFSET:
                continue

            # Remove possible duplicate numerical phase peak at omega = 0
            # before inserting the separately fitted one.
            filtered_positions = []
            filtered_weights = []

            for position, weight in zip(__phase_peak_positions[i], __phase_peak_weights[i]):
                if not is_phase_peak(position, self.uses_coulomb(i)):
                    filtered_positions.append(position)
                    filtered_weights.append(weight)

            __phase_peak_positions[i] = filtered_positions
            __phase_peak_weights[i] = filtered_weights

            # Real part should not have an imaginary shift.
            # This yields a cleaner 1/x form instead of x/(x^2 + delta^2).
            __phase_real = lambda x, res=res: res.continued_fraction(
                x,
                "phase_SC"
            ).real

            # Imaginary part needs an imaginary shift to resolve delta peaks.
            __phase_imag = lambda x, res=res: res.continued_fraction(
                x + FIT_COMPLEX_SHIFT,
                "phase_SC"
            ).imag

            __phase_result = self.fit_goldstone_peak(
                __phase_real,
                __phase_imag,
                i
            )

            __phase_peak_positions[i].insert(0, __phase_result.position)
            __phase_peak_weights[i].insert(0, __phase_result.weight)

        __phase_peak_positions = [
            np.array(positions)
            for positions in __phase_peak_positions
        ]

        __phase_peak_weights = [
            np.array(weights)
            for weights in __phase_peak_weights
        ]

        return __phase_peak_positions, __phase_peak_weights

    def compute_peaks(self):
        """
        Wrapper matching the old return structure:

            higgs_positions, higgs_weights, phase_positions, phase_weights

        All positions are in eV.
        """
        __higgs_peak_positions, __higgs_peak_weights = self.compute_higgs_peaks()
        __phase_peak_positions, __phase_peak_weights = self.compute_phase_peaks()

        return (
            __higgs_peak_positions,
            __higgs_peak_weights,
            __phase_peak_positions,
            __phase_peak_weights
        )

    def __remove_data_below_continuum__(self, spectral_functions):
        if not self.scale_energy_by_gaps:
            for i in range(self.N_data):
                spectral_functions[:, i][self.y < self.true_gaps[i] - CONTINUUM_CUT_SHIFT] = 0
        else:
            for i in range(self.N_data):
                spectral_functions[:, i][self.y * self.max_gaps[i] < self.true_gaps[i] - CONTINUUM_CUT_SHIFT] = 0

    def plot(self, axes, cmap, cbar_max = CBAR_MAX, labels=True):
        spectral_functions_higgs = np.array([res.spectral_density(self.__to_eV__(self.y, __i) + 1e-4j, "amplitude_SC") for __i, res in enumerate(self.resolvents)]).transpose()
        spectral_functions_phase = np.array([res.spectral_density(self.__to_eV__(self.y, __i) + 1e-4j, "phase_SC")     for __i, res in enumerate(self.resolvents)]).transpose()

        if not self.scale_energy_by_gaps:
            (__higgs_peak_positions, __higgs_peak_weights,
                __phase_peak_positions, __phase_peak_weights) = self.compute_peaks()

            self.HiggsModes = pd.DataFrame([ {
                    "resolvent_type": "Higgs",
                    "energies": MEV * __higgs_peak_positions[i],
                    "weights": __higgs_peak_weights[i],
                    "Delta_max": self.data_frame["Delta_max"].iloc[i],
                    "true_gap": self.true_gaps[i],
                    "g": self.data_frame["g"].iloc[i],
                    "error_g": self.__get_error__("g", i),
                    "error_Delta_max": self.__get_error__("Delta_max", i),
                    "error_true_gap": self.__get_error__("true_gap", i),
                    "T": self.data_frame["T"].iloc[i],
                    "omega_D": self.data_frame["omega_D"].iloc[i],
                    "E_F": self.data_frame["E_F"].iloc[i],
                    "k_F": self.data_frame["k_F"].iloc[i],
                    "lambda_screening": self.data_frame["lambda_screening"].iloc[i],
                    "coulomb": self.uses_coulomb(i)
                } for i in range(self.N_data) ])

            self.PhaseModes = pd.DataFrame([ {
                    "resolvent_type": "Phase",
                    "energies": MEV * __phase_peak_positions[i],
                    "weights": __phase_peak_weights[i],
                    "Delta_max": self.data_frame["Delta_max"].iloc[i],
                    "true_gap": self.true_gaps[i],
                    "g": self.data_frame["g"].iloc[i],
                    "error_g": self.__get_error__("g", i),
                    "error_Delta_max": self.__get_error__("Delta_max", i),
                    "error_true_gap": self.__get_error__("true_gap", i),
                    "T": self.data_frame["T"].iloc[i],
                    "omega_D": self.data_frame["omega_D"].iloc[i],
                    "E_F": self.data_frame["E_F"].iloc[i],
                    "k_F": self.data_frame["k_F"].iloc[i],
                    "lambda_screening": self.data_frame["lambda_screening"].iloc[i],
                    "coulomb": self.uses_coulomb(i)
                } for i in range(self.N_data)])

            self.__remove_data_below_continuum__(spectral_functions_higgs)
            self.__remove_data_below_continuum__(spectral_functions_phase)

            ## Note, that the phase peak at omega=0 is the derivative of a delta peak
            ## while the other peaks below the continuum are proper delta peaks
            for i in range(self.N_data):
                for peak_position, weight in zip(__higgs_peak_positions[i], __higgs_peak_weights[i]):
                    if is_phase_peak(peak_position, self.uses_coulomb(i)):
                        summand = -weight * derivative_gaussian_bell(self.__to_eV__(self.y, i), 0)
                    else:
                        summand =  weight * gaussian_bell(self.__to_eV__(self.y, i), peak_position)
                    #mask = summand > 1e-4
                    spectral_functions_higgs[:, i] += summand
                for peak_position, weight in zip(__phase_peak_positions[i], __phase_peak_weights[i]):
                    if is_phase_peak(peak_position, self.uses_coulomb(i)):
                        summand = -weight * derivative_gaussian_bell(self.__to_eV__(self.y, i), 0)
                    else:
                        summand =  weight * gaussian_bell(self.__to_eV__(self.y, i), peak_position)
                    #mask = summand > 1e-4
                    spectral_functions_phase[:, i] += summand
        # endif not self.scale_energy_by_gaps

        levels = np.linspace(0, (1.01 * cbar_max)**(1./CBAR_EXP), 255, endpoint=True)**CBAR_EXP
        cnorm = mcolors.PowerNorm(gamma=1/CBAR_EXP, vmin=0, vmax=1.01 * cbar_max)
        
        contour_higgs = axes[0].contourf(self.x, self.y, spectral_functions_higgs, cmap=cmap, levels=levels, norm=cnorm, extend='both', zorder=-20)
        contour_phase = axes[1].contourf(self.x, self.y, spectral_functions_phase, cmap=cmap, levels=levels, norm=cnorm, extend='both', zorder=-20)
        
        for ax in axes:
            if not self.scale_energy_by_gaps:
                ax.plot(self.x, self.true_gaps, color="cyan", ls=":")
            ax.set_rasterization_zorder(-10)
            ax.set_ylim(0., max(self.y))
            ax.set_xscale(self.xscale)
            ax.set_yscale(self.yscale)

        if labels:
            if self.scale_energy_by_gaps:
                axes[0].set_ylabel(legend(r"\omega / (2 \Delta_\mathrm{max})"))
                axes[1].set_ylabel(legend(r"\omega / (2 \Delta_\mathrm{max})"))
            else:
                axes[0].set_ylabel("Higgs\n" + legend(r"\omega", r"meV"))
                axes[1].set_ylabel("Phase\n" + legend(r"\omega", r"meV"))
        axes[1].set_xlabel(self.xlabel)

        return contour_higgs
    
def create_plot(tasks, xscale="linear", scale_energy_by_gaps=False, cmap='inferno', cbar_max=CBAR_MAX, 
                energy_range=None, fig=None, axes=None, touch_panels=True, white_spines=False, height_to_width_ratio=0.6, **additional_fig_kwargs):
    if energy_range is None:
        energy_range = (-0.25, 59.) if not scale_energy_by_gaps else (0., 1.95)
    if fig is None:
        assert(axes is None)
        __figkwargs = { "nrows": 2, "ncols": len(tasks), "sharex": True, "sharey": True, "height_to_width_ratio": height_to_width_ratio }
        fig, axes = create_large_figure(**__figkwargs, **additional_fig_kwargs) if len(tasks) > 2 else create_normal_figure(**__figkwargs, **additional_fig_kwargs)
        
    plotters = []
    if len(tasks) > 1 :
        if len(tasks) > 3:
            for i, axs in enumerate(axes):
                for j, ax in enumerate(axs):
                    ax.annotate(
                        f"({string.ascii_lowercase[i + 2 * (j // 2)]}.{(j % 2) + 1})",
                        xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                        verticalalignment='top', fontfamily='serif', color="white")
        else:
            for i, axs in enumerate(axes):
                for j, ax in enumerate(axs):
                    ax.annotate(
                        f"({string.ascii_lowercase[i]}.{j+1})",
                        xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                        verticalalignment='top', fontfamily='serif', color="white")

        for i, (data_query, x_column, xlabel) in enumerate(tasks):
            plotters.append(HeatmapPlotter(data_query, x_column, xlabel=xlabel, xscale=xscale, 
                                           energy_range=energy_range, scale_energy_by_gaps=scale_energy_by_gaps))
            contour_for_colorbar = plotters[-1].plot(axes[:, i], labels=not bool(i), cmap=cmap, cbar_max=cbar_max)
    else:
        for i, ax in enumerate(axes):
            ax.annotate(
                f"({string.ascii_lowercase[i]})",
                xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                verticalalignment='top', fontfamily='serif', color="white")
        
        for i, (data_query, x_column, xlabel) in enumerate(tasks):
            plotters.append(HeatmapPlotter(data_query, x_column, xlabel=xlabel, xscale=xscale, 
                                           energy_range=energy_range, scale_energy_by_gaps=scale_energy_by_gaps))
            contour_for_colorbar = plotters[-1].plot(axes[:], labels=not bool(i), cmap=cmap, cbar_max=cbar_max)

    
    from matplotlib.ticker import MaxNLocator
    cbar = fig.colorbar(contour_for_colorbar, ax=axes.ravel(), 
                        orientation='vertical', 
                        shrink=0.985, 
                        pad=0.025, 
                        extend='max', 
                        ticks=[0, 1, 5, 10])
    cbar.set_label(legend(r'\mathcal{A}', r'eV', -1))

    if white_spines:
        for ax in axes.ravel():
            ax.tick_params(which='both', colors="white")
            for label in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
                label.set_color("black")
            for spine in ax.spines.values():
                spine.set_color("white")

    if touch_panels:
        if hasattr(axes[0], "__len__"):
            make_panels_touch(fig, axes)
        else:
            make_panels_touch(fig, axes, touch_x=True, touch_y=False)

    return fig, axes, plotters, cbar