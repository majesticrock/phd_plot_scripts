from mrock.get_data import DataLoader, nickel_cut_params
import os
import pandas as pd

FLOW_FILE_NAME = "flow"

def load_full_flow_file(subdir, L, T, U_0, tprime, E_F, force_json=False):
    data_loader = DataLoader()
    file_with_out_ending = data_loader._to_path("nickel_cut", subdir, **nickel_cut_params(L, T, U_0, tprime, E_F))
    JSON_FILE = os.path.join(file_with_out_ending, f"{FLOW_FILE_NAME}.json.gz")
    PKL_FILE  = os.path.join(file_with_out_ending, f"{FLOW_FILE_NAME}.pkl")
    
    if os.path.isfile(PKL_FILE) and not force_json:
        PKL_TIME = os.path.getmtime(PKL_FILE)
        JSON_TIME = os.path.getmtime(JSON_FILE)
        if PKL_TIME > JSON_TIME:
            print(f"Loading pickle {PKL_FILE}...")
            return pd.read_pickle(PKL_FILE)

    print(f"Loading json {JSON_FILE}...")
    data = data_loader.load_panda_file(JSON_FILE)
    data.to_pickle(PKL_FILE)
    
    return data
