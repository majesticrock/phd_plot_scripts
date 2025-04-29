import os
import tkinter as tk
from tkinter import filedialog, ttk
from collections import defaultdict

import pandas as pd

import __path_appender
__path_appender.append()
from get_data import *
import current_density_fourier
import current_density_time

class ParamSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parameter Selector")

        self.df_paths = pd.DataFrame()
        self.param_order = []
        self.dropdowns = {}
        self.dd_vars = {}

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
        
        laser_box = ttk.Combobox(self, textvariable=self.system_selector_vars['laser'], state="readonly", values=["cosine_laser", "continuous_laser"])
        laser_box.grid(row=1, column=1, padx=5, pady=5)
        #laser_box.current(0)
        
        system_box = ttk.Combobox(self, textvariable=self.system_selector_vars['system'], state="readonly", values=["Dirac", "PiFlux"])
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
            self.dd_vars[param].trace_add("write", make_callback(param, self.dd_vars[param]))

            dropdown = ttk.Combobox(self, textvariable=self.dd_vars[param], state="readonly")
            dropdown.grid(row=i+2, column=1, padx=5, pady=5)

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
        
        # Build selection from dropdowns
        selection = {param: self.dd_vars[param].get() for param in self.param_order if self.dd_vars[param].get()}
    
        # Filter paths according to current valid selection
        df_filtered = self.df_paths.copy()
        for param, val in selection.items():
            df_filtered = df_filtered[df_filtered[param] == val]
    
        # Update each dropdown's value list
        for param in self.param_order:
            all_values = sorted(self.df_paths[param].dropna().unique())
            valid_values = sorted(df_filtered[param].dropna().unique())
    
            current_val = self.dd_vars[param].get()
            if current_val:
                self.dropdowns[param]['values'] = all_values
            else:
                self.dropdowns[param]['values'] = valid_values

    def name_type(self):
        temp = [self.base_dir]
        if self.system_selector_vars['data_set'].get() != '':
            temp.append(self.system_selector_vars['data_set'].get())
        if self.system_selector_vars['laser'].get() != '':
            temp.append(self.system_selector_vars['laser'].get())
        if self.system_selector_vars['system'].get() != '':
            temp.append(self.system_selector_vars['system'].get())
        return os.path.join(*temp)        

    def fft(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return

        # Call your function
        main_df = load_panda(
            "HHG", self.name_type(), "current_density.json.gz",
            **hhg_params(**selected_values)
        )

        current_density_fourier.plot_j(main_df, max_freq=self.max_frequency_scale.get())
        
    def j_time(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return

        # Call your function
        main_df = load_panda(
            "HHG", self.name_type(), "current_density.json.gz",
            **hhg_params(**selected_values)
        )
        current_density_time.plot_j(main_df)

if __name__ == "__main__":
    app = ParamSelector()
    app.mainloop()
