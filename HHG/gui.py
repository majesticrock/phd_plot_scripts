import os
import tkinter as tk
from tkinter import filedialog, ttk

import pandas as pd

import __path_appender
__path_appender.append()
from get_data import *
import current_density_fourier
import current_density_time
import laser_function
import matplotlib.pyplot as plt

class ParamSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parameter Selector")

        self.df_paths = pd.DataFrame()
        self.param_order = []
        self.dropdowns = {}
        self.dd_vars = {}

        self.clear_plots()

        default_path_file = ".default_path.txt"
        if os.path.exists(default_path_file):
            with open(default_path_file, "r") as f:
                default_path = f.read().strip()
        else:
            default_path = os.path.abspath("data/HHG")

        ask_user = False
        if ask_user:
            self.base_dir = filedialog.askdirectory(
                title="Select base directory", initialdir=default_path
            )
            if not self.base_dir:
                self.base_dir = default_path
        else:
            self.base_dir = default_path

        if not self.base_dir:
            self.destroy()
            return
        
        self.create_system_selector()
        self.parse_directory()
        self.create_dropdowns()
        self.create_menu_buttons()
        
        tk.Label(self, text='Max. frequency').grid(row=2, column=2, padx=5, pady=5)
        self.max_frequency_scale = tk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL)
        self.max_frequency_scale.set(40)
        self.max_frequency_scale.grid(row=3, column=2, padx=5, pady=5)

    def create_menu_buttons(self):
        row = len(self.param_order) + 2

        self.change_btn = tk.Button(self, text="Change Directory", command=self.change_directory)
        self.change_btn.grid(row=row, column=0, pady=5)
        
        self.update_btn = tk.Button(self, text="Update Options", command=self.update_options)
        self.update_btn.grid(row=row, column=1, pady=5)

        self.j_time_button = tk.Button(self, text="Plot j(t)", command=self.j_time)
        self.j_time_button.grid(row=row + 1, column=0, pady=5, padx=5)
        
        self.fft_button = tk.Button(self, text="Plot j(w)", command=self.fft)
        self.fft_button.grid(row=row + 1, column=1, pady=5, padx=5)
        
        self.overlay_laser = tk.BooleanVar()
        self.overlay_laser_button = tk.Checkbutton(self, text="Overlay laser?", onvalue=1, offvalue=0, variable=self.overlay_laser)
        self.overlay_laser_button.grid(row=row - 1, column=2, pady=5, padx=5)
        
        self.laser_mode = tk.BooleanVar()
        self.laser_mode_button = tk.Checkbutton(self, text="Electric Field?", onvalue=1, offvalue=0, variable=self.laser_mode)
        self.laser_mode_button.grid(row=row, column=2, pady=5, padx=5)
        
        self.laser_button = tk.Button(self, text="Plot laser", command=self.laser)
        self.laser_button.grid(row=row + 1, column=2, pady=5, padx=5)
        
        self.show_button = tk.Button(self, text="Show Plots", command=self.show_plots)
        self.show_button.grid(row=row + 2, column=0, padx=5, pady=5)

        self.clear_button = tk.Button(self, text="Clear Plots", command=self.clear_plots)
        self.clear_button.grid(row=row + 2, column=2, padx=5, pady=5)

    def change_directory(self):
        new_dir = filedialog.askdirectory(title="Select new base directory", initialdir=self.base_dir)
        if new_dir:
            self.base_dir = new_dir
            
            with open(".default_path.txt", "w") as f:
                f.write(self.base_dir)
            
            self.parse_directory()
            self.create_dropdowns()
            self.create_menu_buttons()

    def parse_directory(self):
        records = []
        walker = os.path.join(self.base_dir, self.name_type())
        print(walker)
        for root, dirs, files in os.walk(walker):
            if any(f.endswith('.json.gz') for f in files):
                rel_path = os.path.relpath(root, walker)
                parts = rel_path.split(os.sep)
                param_dict = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=')
                        param_dict[key] = value
                records.append(param_dict)

        self.df_paths = pd.DataFrame(records)
        self.df_paths.dropna(how='any', inplace=True)
        self.param_order = sorted(self.df_paths.columns.tolist())

    def create_system_selector(self):
        def callback(*args):
            self.parse_directory()
            self.update_options()
            
        self.system_selector_vars = {}
        self.system_selector_vars['data_set'] = tk.StringVar(self, name='data_set')
        self.system_selector_vars['laser'] = tk.StringVar(self, name='laser')
        self.system_selector_vars['system'] = tk.StringVar(self, name='system')
        
        self.system_selector_vars['data_set'].trace_add("write", callback)
        self.system_selector_vars['laser'].trace_add("write", callback)
        self.system_selector_vars['system'].trace_add("write", callback)
        
        tk.Label(self, text='Data Set').grid(row=0, column=0, padx=5, pady=5)
        tk.Label(self, text='Laser').grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self, text='System').grid(row=0, column=2, padx=5, pady=5)
        
        data_dirs = [ d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) ]
        data_box = ttk.Combobox(self, textvariable=self.system_selector_vars['data_set'], state="readonly", values=data_dirs)
        data_box.grid(row=1, column=0, padx=5, pady=5)
        
        laser_box = ttk.Combobox(self, textvariable=self.system_selector_vars['laser'], state="readonly", 
                                 values=["cosine_laser", "continuous_laser", "exp_laser", "expA_laser", "expB_laser", "quench_laser", "powerlaw1_laser"])
        laser_box.grid(row=1, column=1, padx=5, pady=5)
        #laser_box.current(0)
        
        system_box = ttk.Combobox(self, textvariable=self.system_selector_vars['system'], state="readonly", values=["Dirac", "PiFlux", "Honeycomb"])
        system_box.grid(row=1, column=2, padx=5, pady=5)
        #system_box.current(1)

    def create_dropdowns(self):
        def make_callback(p, v):
            previous = [""]  # use a mutable container
            def callback(*args):
                old_value = previous[0]
                previous[0] = v.get()
                self.update_options(args[0], old_value, previous[0])

            return callback
        
        for i, param in enumerate(self.param_order):
            label = tk.Label(self, text=param)
            label.grid(row=i+2, column=0, padx=5, pady=5)

            self.dd_vars[param] = tk.StringVar(self, name=param)        
        
            dropdown = ttk.Combobox(self, textvariable=self.dd_vars[param], state="readonly")
            dropdown.grid(row=i+2, column=1, padx=5, pady=5)
            dropdown.bind("<<ComboboxSelected>>", make_callback(param, self.dd_vars[param]))
            
            self.dropdowns[param] = dropdown

        self.update_options()
        #for param in self.param_order:
        #    self.dropdowns[param].current(0)

    def create_query(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order if self.dropdowns[param].get()}
        query_parts = [f"{param}=='{value}'" for param, value in selected_values.items()]
        query_string = " & ".join(query_parts)
        return query_string

    def update_options(self, changed_param=None, changed_from='', changed_to=''):
        if changed_from != '' and changed_to != '':
            selection = {param: self.dd_vars[param].get() for param in self.param_order if self.dd_vars[param].get()}
            df_filtered = self.df_paths.copy()
            for param, val in selection.items():
                df_filtered = df_filtered[df_filtered[param] == val]
            if len(df_filtered) == 0: 
                for param in self.param_order:
                    if param == changed_param:
                        continue
                    self.dropdowns[param].set('')      
        
        selection = {param: self.dd_vars[param].get() for param in self.param_order if self.dd_vars[param].get()}
    
        df_filtered = self.df_paths.copy()
        for param, val in selection.items():
            df_filtered = df_filtered[df_filtered[param] == val]
    
        for param in self.param_order:
            all_values = sorted(self.df_paths[param].dropna().unique())
            valid_values = sorted(df_filtered[param].dropna().unique())
    
            current_val = self.dd_vars[param].get()
            if current_val:
                self.dropdowns[param]['values'] = all_values
            else:
                self.dropdowns[param]['values'] = valid_values
        
        for key, dropdown in self.dropdowns.items():
            if len(dropdown['values']) == 1:
                dropdown.set(dropdown['values'][0])

    def name_type(self):
        temp = [self.base_dir]
        if self.system_selector_vars['data_set'].get() != '':
            temp.append(self.system_selector_vars['data_set'].get())
        if self.system_selector_vars['laser'].get() != '':
            temp.append(self.system_selector_vars['laser'].get())
        if self.system_selector_vars['system'].get() != '':
            temp.append(self.system_selector_vars['system'].get())
        return os.path.join(*temp)        

    def __get_selected(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return
        return selected_values

    def fft(self):
        if self.frequency_fig is None:
            self.frequency_fig, self.frequency_ax = current_density_fourier.create_frame()

        main_df = load_panda(
            "HHG", self.name_type(), "current_density.json.gz",
            **hhg_params(**self.__get_selected())
        )
        current_density_fourier.add_current_density_to_plot(main_df, self.frequency_ax, label=f"${self.count_freq}$")
        if self.frequencies is None:
            self.frequencies = main_df["frequencies"]
            
        self.count_freq += 1
        
    def j_time(self):
        main_df = load_panda(
            "HHG", self.name_type(), "current_density.json.gz",
            **hhg_params(**self.__get_selected())
        )
        if self.overlay_laser.get():
            if self.time_fig is None:
                self.time_fig, self.time_ax = current_density_time.create_frame()
            current_density_time.add_current_density_to_plot(main_df, self.time_ax, label="$j$")
            if self.laser_ax is None:
                self.laser_ax = self.time_ax.twinx()
            self.laser_ax = self.time_ax.twinx()
            self.laser_ax.set_ylabel("Laser")
            laser_function.add_laser_to_plot(main_df, self.laser_ax, self.laser_mode.get(), color="red")
            
            lower, upper = self.time_ax.get_ylim()
            lower_laser, upper_laser = self.laser_ax.get_ylim()
            
            if -lower > upper:
                self.time_ax.set_ylim(lower, -lower)
            else:
                self.time_ax.set_ylim(-upper, upper)
            
            if -lower_laser > upper_laser:
                self.laser_ax.set_ylim(lower_laser, -lower_laser)
            else:
                self.laser_ax.set_ylim(-upper_laser, upper_laser)
            self.time_ax.axhline(0, c="k", ls="--")
            plt.show()
        else:
            if self.time_fig is None:
                self.time_fig, self.time_ax = current_density_time.create_frame()
            current_density_time.add_current_density_to_plot(main_df, self.time_ax, label=f"${self.count_time}$", normalize=False)
            self.count_time += 1
        
    def laser(self):
        main_df = load_panda(
            "HHG", self.name_type(), "current_density.json.gz",
            **hhg_params(**self.__get_selected())
        )
        if self.laser_ax is None:
            if self.time_ax is None:
                self.time_fig, self.time_ax = current_density_time.create_frame()
            self.laser_ax = self.time_ax.twinx()
        
        laser_function.add_laser_to_plot(main_df, self.laser_ax, self.laser_mode.get(), color="red")

    def show_plots(self):
        if self.frequency_ax is not None:
            self.frequency_ax.set_xlim(0, self.max_frequency_scale.get())
            
            if not self.verticals:
                current_density_fourier.add_verticals(self.frequencies, self.frequency_ax, self.max_frequency_scale.get())
                self.verticals = False
        
        if self.frequency_fig is not None:
            self.frequency_ax.legend()
            #self.frequency_fig.show()
        if self.time_fig is not None:
            self.time_ax.legend()
            #self.time_fig.show()
        plt.show()

    def clear_plots(self):
        self.frequency_fig = None
        self.frequency_ax = None
        self.count_freq = 0
        self.verticals = False
        self.frequencies = None
        
        self.time_fig = None
        self.time_ax = None
        self.laser_ax = None
        self.count_time = 0

if __name__ == "__main__":
    app = ParamSelector()
    app.mainloop()
