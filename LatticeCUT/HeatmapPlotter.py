import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import string
from scipy.signal import find_peaks

import __path_appender as __ap
__ap.append()
import continued_fraction_pandas as cf
import spectral_peak_analyzer as spa
from legend import *

CBAR_MAX = 20
CBAR_EXP = 2

# Settings the importer
BUILD_DIR = "plots/"
FILE_ENDING = ".pdf"
G_MAX_LOAD = 1.5
G_MAX_PLOT = 1.5

__BEGIN__OFFSET__ = 1e-4
__RANGE__ = 1e-5
__SECOND_BEGIN__ = 1e-7
__SECOND_RANGE__ = 1e-6
__PEAK_PROMINCE__ = 0.05

__INITIAL_IMAG__ = 1e-4j
__FIT_COMPLEX_SHIFT__ = 1e-8j
__CONTINUUM_CUT_SHIFT__ = 1e-5

def gaussian_bell(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def derivative_gaussian_bell(x, mu, sigma):
    return -((x - mu) / sigma**2) * gaussian_bell(x, mu, sigma)

def is_phase_peak(peak):
    return abs(peak) < 2e-4

def extract_model_settings(ds):
    return f"dos={ds['dos_name']}  omega_D={ds['omega_D']}  g={ds['g']}  U={ds['U']}  E_F={ds['E_F']}"

class HeatmapPlotter:
    def __init__(self, data_frame_param, parameter_name, xlabel, zlabel=r'$\mathcal{A}(\omega)$', xscale="linear", yscale="linear",
                 energy_range=(1e-10, 2.5), scale_energy_by_gaps=False):
        self.data_frame = data_frame_param.sort_values(parameter_name).reset_index(drop=True)
        
        self.y = np.linspace(energy_range[0], energy_range[1], 5000) # meV
        self.x = (self.data_frame[parameter_name]).to_numpy()
        self.scale_energy_by_gaps = scale_energy_by_gaps
        self.resolvents = [cf.ContinuedFraction(pd_row, messages=False, ignore_first=70, ignore_last=150) for index, pd_row in self.data_frame.iterrows()]
        self.max_gaps   = np.array([2 * gap for gap in self.data_frame["Delta_max"]]) # meV
        self.true_gaps  = np.array([float(t_gap[0]) for t_gap in self.data_frame["continuum_boundaries"]]) # meV
        self.N_data = len(self.max_gaps)
        
        #self.g_cuts = np.zeros(len(DATA_CUTS))
        #for i in range(len(DATA_CUTS)):
        #    filtered_df = self.data_frame[self.data_frame['Delta_max'] < DATA_CUTS[i]]
        #    if len(filtered_df) == 0:
        #        self.g_cuts[i] = 0
        #    else:
        #        closest_row = filtered_df.loc[(DATA_CUTS[i] - filtered_df['Delta_max']).idxmin()]
        #        self.g_cuts[i] = closest_row['g']

        self.xlabel = xlabel
        self.zlabel = zlabel
        self.xscale = xscale
        self.yscale = yscale

    def __scale_if__(self, x, i):
        if self.scale_energy_by_gaps:
            return self.max_gaps[i] * x
        return x

    def identify_modes(self, spectral, pos):
        if self.max_gaps[pos] == 0:
            return np.array([])
        indizes = find_peaks(spectral, prominence=__PEAK_PROMINCE__)[0]
        positions = np.array([self.__scale_if__(self.y[i], pos) for i in indizes])
        positions = positions[positions < self.true_gaps[pos] - __CONTINUUM_CUT_SHIFT__]
        positions[positions < 1e-4] = 0
        return positions

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
        
        #__OFFSET__ = 2
        #if key == "true_gap":
        #    return [self.true_gaps[i]              - self.true_gaps[i - __OFFSET__] if i - __OFFSET__ >= 0                   else None,
        #            self.true_gaps[i + __OFFSET__] - self.true_gaps[i]              if i + __OFFSET__ <  len(self.true_gaps) else None]
        #return [self.data_frame[key].iloc[i]              - self.data_frame[key].iloc[i - __OFFSET__] if i - __OFFSET__ >= 0                         else None,
        #        self.data_frame[key].iloc[i + __OFFSET__] - self.data_frame[key].iloc[i]              if i + __OFFSET__ <  len(self.data_frame["g"]) else None]

    # This function assumes that the quantities provided come in eV
    def __decide_if_to_reverse__(self, peak_idx, peaks, gap):
        if len(peaks) > peak_idx and peaks[peak_idx] - __BEGIN__OFFSET__ - __RANGE__ <= 0:
            return False
        if len(peaks) > peak_idx + 1:
            return abs(peaks[peak_idx + 1] - peaks[peak_idx]) < np.sqrt(__RANGE__)
        else:
            if len(peaks) > peak_idx:
                return abs(peaks[peak_idx] - gap) < np.sqrt(__RANGE__)
            else:
                return False

    def compute_peaks(self, spectral_functions_higgs, spectral_functions_phase):
        __higgs_peak_positions = [ self.identify_modes(spectral_functions_higgs[:, i], i) for i in range(self.N_data) ]
        __higgs_peak_weights   = [ ]
        __higgs_peak_errors    = [ ]
        
        __phase_peak_positions = [ self.identify_modes(spectral_functions_phase[:, i], i) for i in range(self.N_data) ]
        __phase_peak_weights   = [ ]
        __phase_peak_errors    = [ ]

        for i, res in enumerate(self.resolvents):
            # Real part should not have an imaginary shift -> yields better fits
            # because the analytical form is then 1/x rather than x/(x^2 + delta^2)
            __higgs__real = lambda x: res.continued_fraction(x, "amplitude_SC").real
            __phase__real = lambda x: res.continued_fraction(x, "phase_SC").real
            # Imaginary part needs an imaginary shift to resolve the delta peaks
            __higgs__imag = lambda x: res.continued_fraction(x + __FIT_COMPLEX_SHIFT__, "amplitude_SC").imag
            __phase__imag = lambda x: res.continued_fraction(x + __FIT_COMPLEX_SHIFT__, "phase_SC").imag
            
            __higgs_result = [ spa.analyze_peak(__higgs__real, __higgs__imag, 
                                                peak_position         = peak_position, 
                                                range                 = __RANGE__,
                                                begin_offset          = __BEGIN__OFFSET__,
                                                reversed              = self.__decide_if_to_reverse__(__i, __higgs_peak_positions[i], self.true_gaps[i]),
                                                lower_continuum_edge  = self.true_gaps[i]) 
                                for __i, peak_position in enumerate(__higgs_peak_positions[i]) ]
            
            for j, result in enumerate(__higgs_result):
                if abs(result.slope.nominal_value + 1) > 0.05:
                    # Retry the fit with changed parameters
                    __higgs_result[j] = spa.analyze_peak(__higgs__real, __higgs__imag, 
                                                peak_position         = result.position, 
                                                range                 = __SECOND_RANGE__,
                                                begin_offset          = __SECOND_BEGIN__,
                                                reversed              = self.__decide_if_to_reverse__(j, __higgs_peak_positions[i], self.true_gaps[i]),
                                                lower_continuum_edge  = self.true_gaps[i],
                                                improve_peak_position = False) 
                if abs(__higgs_result[j].slope.nominal_value + 1) > 0.05:
                    print("WARNING in Higgs! Expected slope does not match fitted slope!")
                    print(result)
                    print(extract_model_settings(self.data_frame.iloc[i]), "\nReversed=", self.__decide_if_to_reverse__(j, __higgs_peak_positions[i], self.true_gaps[i]), "\n")
            
            __phase_result = [ spa.analyze_peak(__phase__real, __phase__imag, 
                                                peak_position         = peak_position,
                                                range                 = __RANGE__ if not is_phase_peak(peak_position) else __RANGE__ * (1 if self.data_frame["g"].iloc[i] > 0.8 else 50),
                                                begin_offset          = __BEGIN__OFFSET__ if not is_phase_peak(peak_position) else peak_position + __BEGIN__OFFSET__* (1 if self.data_frame["g"].iloc[i] > 0.8 else 50),
                                                reversed              = self.__decide_if_to_reverse__(__i, __phase_peak_positions[i], self.true_gaps[i]),
                                                lower_continuum_edge  = self.true_gaps[i],
                                                ) 
                                for __i, peak_position in enumerate(__phase_peak_positions[i]) ]

            for j, result in enumerate(__phase_result):
                if is_phase_peak(result.position):
                    # The data for small g can be fitted in accordance to expecation
                    # but doing so requires special care becomes the gap is comparatively small, so that the fitting range must be adjusted
                    # Using the general parameters given here yields fits in the range of -2.1 ~ -2.5 rather than the expected -2
                    # nevertheless, the weights are being computed well enough to create a good looking plot
                    if self.data_frame["g"].iloc[i] <= 0.5:
                        continue
                    if abs(result.slope.nominal_value + 2) > 0.2:
                        print("WARNING in Phase! Expected slope of '-2' does not match fitted slope!")
                        print(result)
                        print(extract_model_settings(self.data_frame.iloc[i]), "\nReversed=", self.__decide_if_to_reverse__(j, __phase_peak_positions[i], self.true_gaps[i]), "\n")
                else:
                    if abs(result.slope.nominal_value + 1) > 0.05: 
                        __phase_result[j] = spa.analyze_peak(__phase__real, __phase__imag, 
                                                peak_position         = result.position,
                                                range                 = __SECOND_RANGE__,
                                                begin_offset          = __SECOND_BEGIN__,
                                                reversed              = self.__decide_if_to_reverse__(j, __phase_peak_positions[i], self.true_gaps[i]),
                                                lower_continuum_edge  = self.true_gaps[i],
                                                improve_peak_position = False
                                                ) 
                    if abs(__phase_result[j].slope.nominal_value + 1) > 0.05:
                        print("WARNING in Phase! Expected slope of '-1' does not match fitted slope!")
                        print(result)
                        print(extract_model_settings(self.data_frame.iloc[i]), "\nReversed=", self.__decide_if_to_reverse__(j, __phase_peak_positions[i], self.true_gaps[i]), "\n")
            
            # There are numerical artifacts in the data that have tiny weights which we want to remove here
            # These vanish with increasing numerical accuracy, so they are certainly not physical
            __higgs_result = [result for result in __higgs_result if result.weight >= 5e-6] 
            __phase_result = [result for result in __phase_result if result.weight >= 5e-6 or is_phase_peak(result.position)] 
            
            __higgs_peak_positions[i] =  [ result.position      for result in __higgs_result ]
            __higgs_peak_weights.append( [ result.weight        for result in __higgs_result ])
            __higgs_peak_errors.append(  [ result.weight_error  for result in __higgs_result ])
            
            __phase_peak_positions[i] =  [ result.position      for result in __phase_result ]
            __phase_peak_weights.append( [ result.weight        for result in __phase_result ])
            __phase_peak_errors.append(  [ result.weight_error  for result in __phase_result ])
        
        return (__higgs_peak_positions, __higgs_peak_weights, __higgs_peak_errors, 
                __phase_peak_positions, __phase_peak_weights, __phase_peak_errors)

    def __remove_data_below_continuum__(self, spectral_functions):
        if not self.scale_energy_by_gaps:
            for i in range(self.N_data):
                spectral_functions[:, i][self.y < self.true_gaps[i] - __CONTINUUM_CUT_SHIFT__] = 0
        else:
            for i in range(self.N_data):
                spectral_functions[:, i][self.y * self.max_gaps[i] < self.true_gaps[i] - __CONTINUUM_CUT_SHIFT__] = 0

    def plot(self, axes, cmap, cbar_max = CBAR_MAX, labels=True):
        spectral_functions_higgs = np.array([res.spectral_density(self.__scale_if__(self.y, __i) + __INITIAL_IMAG__, "amplitude_SC") for __i, res in enumerate(self.resolvents)]).transpose()
        spectral_functions_phase = np.array([res.spectral_density(self.__scale_if__(self.y, __i) + __INITIAL_IMAG__, "phase_SC")     for __i, res in enumerate(self.resolvents)]).transpose()

        if not self.scale_energy_by_gaps:
            (__higgs_peak_positions, __higgs_peak_weights, __higgs_peak_errors,
                __phase_peak_positions, __phase_peak_weights, __phase_peak_errors) = self.compute_peaks(spectral_functions_higgs, spectral_functions_phase) 

            self.HiggsModes = pd.DataFrame([ {
                    "resolvent_type": "Higgs",
                    "energies": __higgs_peak_positions[i],
                    "weights": __higgs_peak_weights[i],
                    "weight_errors": __higgs_peak_errors[i],
                    "Delta_max": self.data_frame["Delta_max"].iloc[i],
                    "true_gap": self.true_gaps[i],
                    "g": self.data_frame["g"].iloc[i],
                    "error_g": self.__get_error__("g", i),
                    "error_Delta_max": self.__get_error__("Delta_max", i),
                    "error_true_gap": self.__get_error__("true_gap", i),
                    "omega_D": self.data_frame["omega_D"].iloc[i],
                    "E_F": self.data_frame["E_F"].iloc[i],
                    "U": self.data_frame["U"].iloc[i],
                    "dos_name" : self.data_frame["dos_name"]
                } for i in range(self.N_data) ])
            self.PhaseModes = pd.DataFrame([ {
                    "resolvent_type": "Phase",
                    "energies": __phase_peak_positions[i],
                    "weights": __phase_peak_weights[i],
                    "weight_errors": __phase_peak_errors[i],
                    "Delta_max": self.data_frame["Delta_max"].iloc[i],
                    "true_gap": self.true_gaps[i],
                    "g": self.data_frame["g"].iloc[i],
                    "error_g": self.__get_error__("g", i),
                    "error_Delta_max": self.__get_error__("Delta_max", i),
                    "error_true_gap": self.__get_error__("true_gap", i),
                    "omega_D": self.data_frame["omega_D"].iloc[i],
                    "E_F": self.data_frame["E_F"].iloc[i],
                    "U": self.data_frame["U"].iloc[i],
                    "dos_name" : self.data_frame["dos_name"]
                } for i in range(self.N_data)])

            self.__remove_data_below_continuum__(spectral_functions_higgs)
            self.__remove_data_below_continuum__(spectral_functions_phase)

            sigma = 0.00005
            ## Note, that the phase peak at omega=0 is the derivative of a delta peak
            ## while the other peaks below the continuum are proper delta peaks
            for i in range(self.N_data):
                for peak_position, weight in zip(__higgs_peak_positions[i], __higgs_peak_weights[i]):
                    if is_phase_peak(peak_position):
                        summand = -weight * derivative_gaussian_bell(self.__scale_if__(self.y, i), 0, 4 * sigma)
                    else:
                        summand =  weight * gaussian_bell(self.__scale_if__(self.y, i), peak_position, sigma)
                    mask = summand > 1e-4
                    spectral_functions_higgs[mask, i] += summand[mask]
                for peak_position, weight in zip(__phase_peak_positions[i], __phase_peak_weights[i]):
                    if is_phase_peak(peak_position):
                        summand = -weight * derivative_gaussian_bell(self.__scale_if__(self.y, i), 0, 4 * sigma)
                    else:
                        summand =  weight * gaussian_bell(self.__scale_if__(self.y, i), peak_position, sigma)
                    mask = summand > 1e-4
                    spectral_functions_phase[mask, i] += summand[mask]
        # endif not self.scale_energy_by_gaps

        levels = np.linspace(0, (1.01 * cbar_max)**(1./CBAR_EXP), 101, endpoint=True)**CBAR_EXP
        #levels = 201#np.linspace(0, 1.01 * cbar_max, 201, endpoint=True)

        cnorm = mcolors.PowerNorm(gamma=1/CBAR_EXP, vmin=0, vmax=1.01 * cbar_max)
        
        contour_higgs = axes[0].contourf(self.x, self.y, spectral_functions_higgs, cmap=cmap, levels=levels, norm=cnorm, extend='max', zorder=-20)
        contour_phase = axes[1].contourf(self.x, self.y, spectral_functions_phase, cmap=cmap, levels=levels, norm=cnorm, extend='max', zorder=-20)
        contour_higgs.set_edgecolor('face')
        contour_phase.set_edgecolor('face')
        
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
                axes[0].set_ylabel(legend(r"\omega"))
                axes[1].set_ylabel(legend(r"\omega"))
        axes[1].set_xlabel(self.xlabel)

        return contour_higgs
    
def create_plot(tasks, xscale="linear", scale_energy_by_gaps=False, cmap='inferno', cbar_max=CBAR_MAX, energy_range=None, fig=None, axes=None):
    if energy_range is None:
        energy_range = (0., 0.5) if not scale_energy_by_gaps else (0., 1.95)
    if fig is None:
        assert(axes is None)
        fig, axes = plt.subplots(nrows=2, ncols=len(tasks), figsize=(12.8 if len(tasks) > 2 else 6.4, 4.8), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        
    ticks = [0]
    while ticks[-1] < 0.95 * cbar_max:
        ticks.append(ticks[-1] + 0.1 * cbar_max)
    
    plotters = []
    if len(tasks) > 1 :
        if len(tasks) > 3:
            for i, axs in enumerate(axes):
                for j, ax in enumerate(axs):
                    ax.annotate(
                        f"({string.ascii_lowercase[i + 2 * (j // 2)]}.{(j % 2) + 1})",
                        xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                        verticalalignment='top', color="white", weight="bold")
        else:
            for i, axs in enumerate(axes):
                for j, ax in enumerate(axs):
                    ax.annotate(
                        f"({string.ascii_lowercase[i]}.{j+1})",
                        xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                        verticalalignment='top', color="white", weight="bold")

        for i, (data_query, x_column, xlabel) in enumerate(tasks):
            plotters.append(HeatmapPlotter(data_query, x_column, xlabel=xlabel, xscale=xscale, 
                                           energy_range=energy_range, scale_energy_by_gaps=scale_energy_by_gaps))
            contour_for_colorbar = plotters[-1].plot(axes[:, i], labels=not bool(i), cmap=cmap, cbar_max=cbar_max)
            
        cbar = fig.colorbar(contour_for_colorbar, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.1, pad=0.025, extend='max', ticks=ticks)
    else:
        for i, ax in enumerate(axes):
            ax.annotate(
                f"({string.ascii_lowercase[i]})",
                xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
                verticalalignment='top', color="white", weight="bold")
        
        for i, (data_query, x_column, xlabel) in enumerate(tasks):
            plotters.append(HeatmapPlotter(data_query, x_column, xlabel=xlabel, xscale=xscale, 
                                           energy_range=energy_range, scale_energy_by_gaps=scale_energy_by_gaps))
            contour_for_colorbar = plotters[-1].plot(axes[:], labels=not bool(i), cmap=cmap, cbar_max=cbar_max)

        cbar = fig.colorbar(contour_for_colorbar, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.1, pad=0.025, extend='max', ticks=ticks)
    
    cbar.set_label(legend(r'\mathcal{A}(\omega)'))
    return fig, axes, plotters, cbar