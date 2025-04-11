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

        # Optional dialog, only if you want it
        ask_user = False  # flip this to True to show dialog
        default_path = os.path.abspath("data/HHG/4_cycle/cosine_laser")
        self.name_type = "4_cycle/cosine_laser"

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

        self.parse_directory()
        self.create_dropdowns()
        self.create_menu_buttons()

    def create_menu_buttons(self):
        row = len(self.param_order) + 1

        self.change_btn = tk.Button(self, text="Change Directory", command=self.change_directory)
        self.change_btn.grid(row=row, column=0, columnspan=2, pady=5)

        self.j_time_button = tk.Button(self, text="Plot j(t)", command=self.j_time)
        self.j_time_button.grid(row=row + 1, column=0, pady=5, padx=5)
        
        self.fft_button = tk.Button(self, text="Plot j(w)", command=self.fft)
        self.fft_button.grid(row=row + 1, column=1, pady=5, padx=5)

    def change_directory(self):
        new_dir = filedialog.askdirectory(title="Select new base directory", initialdir=self.base_dir)
        if new_dir:
            self.base_dir = new_dir
            
            path_parts = os.path.normpath(new_dir).split(os.sep)
            if len(path_parts) >= 2:
                self.name_type = os.path.join(path_parts[-2], path_parts[-1])
                print("Selected:", self.name_type)
            else:
                self.name_type = ""
                print("Warning: Could not extract name/type")
            
            self.parse_directory()
            self.create_dropdowns()
            self.create_menu_buttons()

    def parse_directory(self):
        records = []
        for root, dirs, files in os.walk(self.base_dir):
            if any(f.endswith('.json.gz') for f in files):
                rel_path = os.path.relpath(root, self.base_dir)
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
            label.grid(row=i, column=0, padx=5, pady=5)

            self.dd_vars[param] = tk.StringVar(self, name=param)        
            self.dd_vars[param].trace_add("write", make_callback(param, self.dd_vars[param]))

            dropdown = ttk.Combobox(self, textvariable=self.dd_vars[param], state="readonly")
            dropdown.grid(row=i, column=1, padx=5, pady=5)

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

    def fft(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return

        # Call your function
        main_df = load_panda(
            "HHG", self.name_type, "current_density.json.gz",
            **hhg_params(**selected_values)
        )

        current_density_fourier.plot_j(main_df)
        
    def j_time(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return

        # Call your function
        main_df = load_panda(
            "HHG", self.name_type, "current_density.json.gz",
            **hhg_params(**selected_values)
        )
        current_density_time.plot_j(main_df)

if __name__ == "__main__":
    app = ParamSelector()
    app.mainloop()
