import pickle, gzip
import yaml
import json
import numpy as np
import os
import matplotlib as mpl
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_session(data_path, date):
    path = os.path.join(data_path, f"output_{date}.pkl_gz")
    with gzip.open(path, "rb") as f:
        sess = pickle.load(f)
    return sess 

def save_inc(inc):
    with open("./outputs/include.pkl", 'wb') as f:
        pickle.dump(inc, f)

def load_inc():
    with open("./outputs/include.pkl", 'rb') as f:
        inc = pickle.load(f)
    return inc

def save_dirs(inc):
    with open("./outputs/dirs.pkl", 'wb') as f:
        pickle.dump(inc, f)

def load_dirs():
    with open("./outputs/dirs.pkl", 'rb') as f:
        inc = pickle.load(f)
    return inc

def save_ccgs(ccgs, label=""):
    with open(f"./outputs/ccgs{label}.pkl", 'wb') as f:
        pickle.dump(ccgs, f)

def load_ccgs(label=""):
    with open(f"./outputs/ccgs{label}.pkl", 'rb') as f:
        ccgs = pickle.load(f)
    return ccgs

def save_waves(waves):
    with open("./outputs/waves.pkl", 'wb') as f:
        pickle.dump(waves, f)

def load_waves():
    with open("./outputs/waves.pkl", 'rb') as f:
        waves = pickle.load(f)
    return waves

def save_rfs(rfs):
    with open("./outputs/rfs.pkl", 'wb') as f:
        pickle.dump(rfs, f)

def save_video(rfs):
    with open("./outputs/video_metrics.pkl", 'wb') as f:
        pickle.dump(rfs, f)

def load_video():
    with open("./outputs/video_metrics.pkl", 'rb') as f:
        rfs = pickle.load(f)
    return rfs

def load_rfs():
    with open("./outputs/rfs.pkl", 'rb') as f:
        rfs = pickle.load(f)
    return rfs

def save_pat(pat, suffix=None):
    if suffix == None:
        with open("./outputs/pattern.pkl", 'wb') as f:
            pickle.dump(pat, f)
    else:
        with open(f"./outputs/pattern{suffix}.pkl", 'wb') as f:
            pickle.dump(pat, f)        

def load_pat(suffix = None):
    if suffix == None: 
        with open("./outputs/pattern.pkl", 'rb') as f:
            pat = pickle.load(f)
    else:
        with open(f"./outputs/pattern{suffix}.pkl", 'rb') as f:
            pat = pickle.load(f)
    return pat

def save_fractures(fractures, suffix=None):
    if suffix is None or suffix == "":
        path = "./outputs/fractures.pkl"
    else:
        path = f"./outputs/fractures{suffix}.pkl"
    with open(path, "wb") as f:
        pickle.dump(fractures, f)

def load_fractures(suffix=None):
    if suffix is None or suffix == "":
        path = "./outputs/fractures.pkl"
    else:
        path = f"./outputs/fractures{suffix}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_heatmaps(heatmaps, suffix=None):
    if suffix is None or suffix == "":
        path = "./outputs/heatmaps.pkl"
    else:
        path = f"./outputs/heatmaps{suffix}.pkl"
    with open(path, "wb") as f:
        pickle.dump(heatmaps, f)

def load_heatmaps(suffix=None):
    if suffix is None or suffix == "":
        path = "./outputs/heatmaps.pkl"
    else:
        path = f"./outputs/heatmaps{suffix}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def get_pref_task_name_id(data, task_name=None):
    if task_name is None:
        if "plaid_task" in data.keys() and len(data["plaid_task"]) > 0:
            task_name = "plaid_task"
        elif "rf_task" in data.keys() and len(data["rf_task"]) > 0:
            task_name = "rf_task"
        elif "video_task" in data.keys() and len(data["video_task"]) > 0:
            task_name = "video_task"
        else:
            raise ValueError("")  

    if task_name == "plaid_task":
        max_sz = 0
        task_id = 0
        trial_thresh = 100
        was_eq = 0
        any_eq = False
        for i in range(len(data["plaid_task"])): 
            if data["plaid_task"][i]["probe_class"] == "equal_contrast":
                any_eq = True

        for i in range(len(data["plaid_task"])): 
            sz = data["plaid_task"][i]["probe_size"]
            nt = data["plaid_task"][i]['spikes'].shape[0]
            if (sz > max_sz and nt > trial_thresh):
                if any_eq and data["plaid_task"][i]["probe_class"] != "equal_contrast":
                    continue # if there are any equal contrast probes, prefer those sessions
                task_id = i
                max_sz = sz

    elif task_name == "video_task":
        task_name = "video_task"
        most_tri = 0
        task_id = 0
        # default to more trials
        for i in range(len(data["video_task"])): 
            nt = data["video_task"][i]['spikes'].shape[0]
            if nt > most_tri:
                task_id = i
                most_tri = nt    
    elif task_name == "rf_task":
        task_name = "rf_task"
        most_tri = 0
        task_id = 0
        big_sz = 0
        # default to more trials w/ largest probe size
        for i in range(len(data["rf_task"])): 
            nt = data["rf_task"][i]['spikes'].shape[0]
            sz = data["rf_task"][i]["probe_size"]
            if sz > big_sz | (sz == big_sz and nt > most_tri):
                task_id = i
                most_tri = nt        
                big_sz = sz

    return task_name, task_id

def sort_dates_and_make_session_titles(config, split, monkey_order=("A","H","T"), date_fmt="%m%d%y"):
    if split == "all":
        dates_by_monkey = {
            "A": config["dataset_params"]["dates_A"],
            "H": config["dataset_params"]["dates_H"],
        }
    elif split == "main":
        dates_by_monkey = {
            "A": config["dataset_params"]["dates_A_main"],
            "H": config["dataset_params"]["dates_H_main"],
        }
    elif split == "supplement":
        dates_by_monkey = {
            "A": config["dataset_params"]["dates_A_supplement"],
            "H": config["dataset_params"]["dates_H_supplement"],
        }

    ordered_dates=[]
    date_to_session={}
    for m in monkey_order:
        if m not in dates_by_monkey: 
            continue
        ds=sorted(dates_by_monkey[m], key=lambda d: datetime.strptime(d, date_fmt), reverse=True)
        for i,d in enumerate(ds,1): 
            ordered_dates.append(d)
            date_to_session[d]=f"{m}, Session {i}"
    return ordered_dates, date_to_session

def set_plot_defaults():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    pattern_color = "tab:orange"
    component_color = "tab:blue"
    unclass_color = "#888888"

    return pattern_color, component_color, unclass_color

def save_date_task_map(date_task_map):
    with open("./outputs/date_task_map.json", 'w') as f:
        json.dump(date_task_map, f, indent=2)

def load_date_task_map():
    with open("./outputs/date_task_map.json", 'r') as f:
        date_task_map = json.load(f)
    return date_task_map

def save_rf_task_map(rf_task_map):
    with open("./outputs/rf_task_map.json", 'w') as f:
        json.dump(rf_task_map, f, indent=2)

def load_rf_task_map():
    with open("./outputs/rf_task_map.json", 'r') as f:
        rf_task_map = json.load(f)
    return rf_task_map