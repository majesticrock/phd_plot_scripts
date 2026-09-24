from mrock.get_data import DataLoader, nickel_cut_params
import os
import re
import pandas as pd

FLOW_FILE_NAME = "flow"
FULL_STATE_FILE_NAME = "full_flow_state"

BINARY_DIR = "binaries"

# The first output files are simply called "FLOW_FILE_NAME.ENDING"
# Simulations resumed from a state are then numbered via "FLOW_FILE_NAME.ENDING{number}" starting with 1
def load_full_flow_file(subdir, L, T, U_0, tprime, E_F, resume_num="", force_json=False):
    data_loader = DataLoader()
    JSON_FILE = os.path.join(data_loader._to_path(
        "nickel_cut", subdir, **nickel_cut_params(L, T, U_0, tprime, E_F)
    ), f"{FLOW_FILE_NAME}.json.gz{resume_num}")
    PKL_FILE = os.path.join(data_loader._to_path(
        f"nickel_cut", os.path.join(subdir, BINARY_DIR), **nickel_cut_params(L, T, U_0, tprime, E_F)
    ), f"{FLOW_FILE_NAME}.pkl{resume_num}")

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

def load_all_resumed_files(subdir, L, T, U_0, tprime, E_F, force_json=False):
    data_loader = DataLoader()
    file_with_out_ending = data_loader._to_path(
        "nickel_cut", subdir, **nickel_cut_params(L, T, U_0, tprime, E_F)
    )

    cache_file = os.path.join(data_loader._to_path(
        f"nickel_cut", os.path.join(subdir, BINARY_DIR), **nickel_cut_params(L, T, U_0, tprime, E_F)
    ), "flow_all.pkl")

    # Match: flow.json.gz, flow.json.gz1, flow.json.gz2, ...
    pattern = re.compile(rf"^{FLOW_FILE_NAME}\.json\.gz(\d*)$")

    json_files = []
    for fname in os.listdir(file_with_out_ending):
        m = pattern.match(fname)
        if m:
            suffix = m.group(1)
            resume_num = int(suffix) if suffix else 0
            json_files.append((resume_num, os.path.join(file_with_out_ending, fname)))

    if not json_files:
        return []

    json_files.sort(key=lambda x: x[0])

    # Use cached list if it is newer than every JSON file
    if os.path.isfile(cache_file) and not force_json:
        cache_time = os.path.getmtime(cache_file)
        newest_json = max(os.path.getmtime(path) for _, path in json_files)

        if cache_time > newest_json:
            print(f"Loading pickle {cache_file}...")
            return pd.read_pickle(cache_file)

    # Load all JSON files
    dataframes = []
    for _, json_path in json_files:
        print(f"Loading json {json_path}...")
        dataframes.append(data_loader.load_panda_file(json_path))

    # Cache the complete list
    pd.to_pickle(dataframes, cache_file)

    return dataframes

def load_full_flow_state(subdir, L, T, U_0, tprime, E_F, resume_num="", force_json=False):
    data_loader = DataLoader()
    JSON_FILE = os.path.join(data_loader._to_path(
        "nickel_cut", subdir, **nickel_cut_params(L, T, U_0, tprime, E_F)
    ), f"{FULL_STATE_FILE_NAME}.json.gz{resume_num}")
    PKL_FILE = os.path.join(data_loader._to_path(
        f"nickel_cut", os.path.join(subdir, BINARY_DIR), **nickel_cut_params(L, T, U_0, tprime, E_F)
    ), f"{FULL_STATE_FILE_NAME}.pkl{resume_num}")
    
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