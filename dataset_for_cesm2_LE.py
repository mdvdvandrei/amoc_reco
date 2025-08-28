from __future__ import annotations

import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import zarr

class PreprocessedCMIP6Dataset_LE(Dataset):
    def __init__(
        self,
        zarr_dir,
        models,
        x_vars,
        scenarios,
        *,
        target_group="y",          # "y" | "y_no_ekman" | "ekman"
        output_type="max",
        selected_lats=None,
        transform=None,
        lpf="raw",
        local=False,
        noise=None,
        member_selection="all"     # NEW: "all" | "first" | list of member IDs
    ):
        assert target_group in {"y", "y_no_ekman", "ekman"}
        self.target_group = target_group

        # storage -----------------------
        self.zarr_dir, self.models, self.x_vars = zarr_dir, models, x_vars
        self.scenarios, self.output_type = scenarios, output_type
        self.selected_lats, self.transform = selected_lats, transform
        self.lpf, self.local = lpf, local
        self.noise = noise
        self.member_selection = member_selection
        
        # sampling stride -----------------------------------
        self.num_skip = 64
        self.stride = 12 if lpf in ("LPF120") else 12

        # ----------------------------------------------------------------
        # global & per‑model norm‑stats (load from previous dataset structure)
        # ----------------------------------------------------------------
        self.global_norm_stats = {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}}
        self.norm_stats = {m: {"x": {}, "y": {}, "y_no_ekman": {}, "ekman": {}} for m in models}

        def _load_stats(model_name, tgt_dict):
            """Load normalization stats from previous dataset structure."""
            path = os.path.join("/home/am334/link_am334/moc_mmodel/monthly_1deg_grid_with_coasts_EK_minus", f"output_data_{model_name}.zarr")
            try:
                root = zarr.open_group(path, mode="r")
            except Exception:
                return
            if "piControl_stats" not in root:
                return
            for grp in ("x", "y", "y_no_ekman", "ekman"):
                if grp not in root["piControl_stats"]:
                    continue
                for key in root["piControl_stats"][grp]:
                    mu = zarr.open_array(os.path.join(path, "piControl_stats", grp, key, "mean"))
                    sd = zarr.open_array(os.path.join(path, "piControl_stats", grp, key, "std"))
                    tgt_dict[grp][key] = (np.asarray(mu), np.asarray(sd))

        # Load global stats from MPI-ESM1-2-HR (as reference model)
        _load_stats("MPI-ESM1-2-HR", self.global_norm_stats)
        # Load per-model stats for all models
        for m in models:
            _load_stats(m, self.norm_stats[m])

        # ----------------------------------------------------------------
        # build self.samples (now including member dimension)
        # ----------------------------------------------------------------
        self.samples = []
        for model in models:
            path = os.path.join(zarr_dir, f"output_{model}.zarr")  # Fixed filename
            try:
                root = zarr.open_group(path, mode="r")
                #print(f"\n=== Model: {model} ===")
                #print(f"Root groups: {list(root.keys())}")
            except Exception as e:
                #print(f"Could not open {path}: {e}")
                continue

            for scn in scenarios:
                if scn not in root:
                    print(f"Scenario {scn} not found in {model}")
                    continue
                
                #print(f"\n--- Scenario: {scn} ---")
                scenario_group = root[scn]
                available_members = [key for key in scenario_group.keys() 
                                   if isinstance(scenario_group[key], zarr.hierarchy.Group)]
                #print(f"Available members: {available_members}")
                
                # Filter members based on selection criteria
                if self.member_selection == "first":
                    selected_members = available_members[:1] if available_members else []
                elif self.member_selection == "all":
                    selected_members = available_members
                elif isinstance(self.member_selection, (list, tuple)):
                    selected_members = [m for m in available_members if m in self.member_selection]
                else:
                    selected_members = available_members
                
                #print(f"Selected members: {selected_members}")
                
                for member_id in selected_members:
                    #print(f"\n  Member: {member_id}")
                    member_group = scenario_group[member_id]
                    #print(f"  Member structure: {list(member_group.keys())}")
                    
                    # Check if 'x' group exists under member
                    if "x" not in member_group:
                        print(f"  No 'x' group found in {member_id}")
                        continue
                    
                    x_group = member_group["x"]
                    #print(f"  X variables: {list(x_group.keys())}")
                    
                    # Check if required variables exist directly in x group (flat structure)
                    x_vars_available = True
                    time_lengths = []
                    
                    for var in x_vars:
                        # Look for variable_filter directly in x group
                        var_key = f"{var}_{lpf}"
                        if var_key not in x_group:
                            print(f"  Variable {var_key} not found in {member_id}/x")
                            x_vars_available = False
                            break
                        
                        var_array = x_group[var_key]
                        time_lengths.append(var_array.shape[0])
                        #print(f"    Found {var_key} with {var_array.shape[0]} timesteps")
                    
                    if not x_vars_available:
                        print(f"  Skipping {member_id} - missing x variables")
                        continue
                    
                    # Check if target group exists
                    if self.target_group not in member_group:
                        print(f"  Target group {self.target_group} not found in {member_id}")
                        continue
                    
                    yg = member_group[self.target_group]
                    #print(f"  Y group structure: {list(yg.keys())}")
                    
                    if lpf not in yg:
                        print(f"  Required filter {lpf} not found in target group")
                        continue
                    
                    time_lengths.append(yg[lpf].shape[0])
                    T = min(time_lengths)
                    #print(f"  Time dimension: {T}, will create {len(range(self.num_skip, T, self.stride))} samples")
                    
                    for t in range(self.num_skip, T, self.stride):
                        self.samples.append({
                            "model": model, 
                            "scenario": scn, 
                            "member": member_id,
                            "time_idx": t
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model = sample["model"]
        scenario = sample["scenario"]
        member = sample["member"]
        time_idx = sample["time_idx"]

        store_path = os.path.join(self.zarr_dir, f"output_{model}.zarr")  # Fixed filename

        # ------------- X -------------------------------------------------
        x_data_list = []
        for var in self.x_vars:
            # Look for variable_filter directly in x group (flat structure)
            var_key = f"{var}_{self.lpf}"
            x_arr_path = os.path.join(store_path, scenario, member, "x", var_key)
            
            try:
                x_arr = zarr.open_array(x_arr_path, mode="r")
            except Exception as e:
                raise ValueError(f"Could not find {var_key} in {model}/{scenario}/{member}/x: {e}")
            
            x_data = np.asarray(x_arr[time_idx]).astype(np.float32)

            # Apply normalization (using stats from previous dataset structure)
            if model in self.norm_stats and f"{var}_raw" in self.norm_stats[model]["x"]:
                norm_mean, norm_std = self.norm_stats[model]["x"][f"{var}_raw"]
                
                if self.noise is not None:
                    x_data = (x_data - norm_mean) #+ np.random.normal(loc=0.0, scale=self.noise, size=x_data.shape))
                else:
                    x_data = (x_data - norm_mean)

            # Handle specific model quirks
            if model == "MRI-ESM2-0":
                x_data[50, 94] = 0

            x_data = np.nan_to_num(x_data)
            x_data_list.append(x_data)

        x_data_concat = np.stack(x_data_list, axis=0)

        # ------------- Y‑like target  -----------------------------------
        y_arr_path = os.path.join(store_path, scenario, member, self.target_group, self.lpf)
        y_arr = zarr.open_array(y_arr_path, mode="r")
        y_data = np.asarray(y_arr[time_idx]).astype(np.float32)

        # Apply normalization for target
        if (model in self.norm_stats and 
            self.lpf in self.norm_stats[model][self.target_group]):
            norm_mean, norm_std = self.norm_stats[model][self.target_group][self.lpf]
            max_idx = np.argmax(y_data, axis=0)

            ref = norm_mean[np.argmax(norm_mean, axis=0), np.arange(norm_mean.shape[1])]
            y_data = (y_data - ref) / 1161159294



        if self.selected_lats is not None:
            lat_arr  = np.arange(-30, 70 + 2.5, 2.5)
            profiles, idx_list = [], []
            for sel in self.selected_lats:
                idx_lat = np.abs(lat_arr - sel).argmin()
                profiles.append(y_data[:, idx_lat])
                idx_list.append(idx_lat)
            y_data = np.stack(profiles, axis=1)

        if self.transform:
            x_data_concat = self.transform(x_data_concat)

        if self.output_type == "max":
            y_data = y_data[max_idx[idx_list], np.arange(y_data.shape[1])]
            #y_data = y_data[1, np.arange(y_data.shape[1])]

        y_data = np.nan_to_num(y_data)
        return torch.from_numpy(x_data_concat), torch.from_numpy(y_data)





    def get_sample_info(self, idx):
        """Return metadata about a specific sample."""
        sample = self.samples[idx]
        return {
            "model": sample["model"],
            "scenario": sample["scenario"], 
            "member": sample["member"],
            "time_idx": sample["time_idx"]
        }

    def get_members_for_model_scenario(self, model, scenario):
        """Get all available members for a given model/scenario combination."""
        members = set()
        for sample in self.samples:
            if sample["model"] == model and sample["scenario"] == scenario:
                members.add(sample["member"])
        return sorted(list(members))


# ================================================================
# EXAMPLE USAGE


if __name__ == "__main__":

    dataset = PreprocessedCMIP6Dataset_LE(
        zarr_dir="/home/am334/link_am334/moc_mmodel/monthly_stream_zarr",
        models=["CESM2"],
        x_vars=["tos"],
        scenarios=["historical"],
        member_selection="all",
        output_type="max",
        target_group = "y",
        lpf          = "LPF24",
        selected_lats = [26.5]

    )

    print("dataset len: ", len(dataset))


    x_tensor, y_tensor = dataset[43]
    print(y_tensor)
    print("x_tensor shape:", x_tensor.shape)  # Expected shape: [num_vars, ...]
    print("y_tensor shape:", y_tensor.shape)

