from mrock.get_data import DataLoader
import os
import pandas as pd

def load_full_flow_file(file_with_out_ending):
    PKL_FILE = file_with_out_ending + ".pkl"
    JSON_FILE = file_with_out_ending + ".json.gz"
    if os.path.isfile(PKL_FILE):
        PKL_TIME = os.path.getmtime(PKL_FILE)
        JSON_TIME = os.path.getmtime(JSON_FILE)
        if PKL_TIME > JSON_TIME:
            print(f"Loading pickle {PKL_FILE}...")
            return pd.read_pickle(PKL_FILE)

    print(f"Loading json {JSON_FILE}...")
    data_loader = DataLoader()
    data = data_loader.load_panda_file(JSON_FILE)
    data.to_pickle(PKL_FILE)
    
    return data
