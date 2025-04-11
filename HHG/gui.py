import os
import tkinter as tk
from tkinter import filedialog, ttk
from collections import defaultdict

import threading

import __path_appender
__path_appender.append()
from get_data import *
from current_density_fourier import plot_j

class ParamSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parameter Selector")

        self.param_tree = defaultdict(set)
        self.valid_paths = []

        self.param_order = []
        self.current_selection = {}
        self.dropdowns = {}

        # Optional dialog, only if you want it
        ask_user = True  # flip this to True to show dialog
        default_path = os.path.abspath("data/HHG/4_cycle/cosine_laser")
        print(default_path)
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

        # Add "Load Data" button
        self.load_button = tk.Button(self, text="Load Data", command=self.load_data)
        self.load_button.grid(row=len(self.param_order), column=0, columnspan=2, pady=10)

    def parse_directory(self):
        for root, dirs, files in os.walk(self.base_dir):
            if any(f.endswith('.json.gz') for f in files):
                rel_path = os.path.relpath(root, self.base_dir)
                parts = rel_path.split(os.sep)
                param_dict = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=')
                        param_dict[key] = value
                self.valid_paths.append(param_dict)

        all_params = set()
        for path in self.valid_paths:
            all_params.update(path.keys())
        self.param_order = sorted(all_params)

    def create_dropdowns(self):
        for i, param in enumerate(self.param_order):
            label = tk.Label(self, text=param)
            label.grid(row=i, column=0, padx=5, pady=5)

            var = tk.StringVar(self)
            var.trace_add("write", lambda *_: self.update_options())

            dropdown = ttk.Combobox(self, textvariable=var, state="readonly")
            dropdown.grid(row=i, column=1, padx=5, pady=5)

            self.dropdowns[param] = dropdown
            self.current_selection[param] = None

        self.update_options()

    def update_options(self):
        selection = {param: self.dropdowns[param].get() for param in self.param_order if self.dropdowns[param].get()}

        filtered_paths = self.valid_paths
        for param, val in selection.items():
            filtered_paths = [p for p in filtered_paths if p.get(param) == val]

        for param in self.param_order:
            possible_values = sorted({p[param] for p in filtered_paths if param in p})
            current = self.dropdowns[param].get()
            self.dropdowns[param]['values'] = possible_values
            if current not in possible_values:
                self.dropdowns[param].set('')

    def load_data(self):
        selected_values = {param: self.dropdowns[param].get() for param in self.param_order}
        if not all(selected_values.values()):
            print("Please make sure all parameters are selected.")
            return

        # Call your function
        main_df = load_panda(
            "HHG", "4_cycle/cosine_laser", "current_density.json.gz",
            **hhg_params(**selected_values)
        )

        plot_j(main_df)

if __name__ == "__main__":
    app = ParamSelector()
    app.mainloop()
