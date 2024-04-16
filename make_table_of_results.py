import argparse
import json
from pathlib import Path

import pandas as pd

# argparser to get the experiment name
parser = argparse.ArgumentParser(description="Create a table from JSON files")
parser.add_argument("--experiment_name", default=None, type=str, nargs="+", help="list of experiments to include.")
args = parser.parse_args()

# Create a Path object with the folder path
folder_path = Path("outputs")

# List to hold all json data
json_list = []
experiment_names_list = []
# Iterate over each JSON file in the directory
for json_file in folder_path.glob("*/*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        # Load the JSON content
        json_content = json.load(f)
        # Append the content as is (which will result in a single cell containing the JSON in the DataFrame)
        experiment_names_list.append(json_content["cfg_experiment_name"])
        if args.experiment_name is not None and json_content["cfg_experiment_name"] in args.experiment_name:
            json_list.append(json_content)
        elif args.experiment_name is None:
            json_list.append(json_content)

print(f"The set of experiment names in the folder is: {set(experiment_names_list)}")
# Convert the list of JSON objects to a DataFrame
df = pd.DataFrame(json_list)
df = df.sort_values(by=["cfg_model_name", "cfg_dataset", "cfg_data_idx", "cfg_discrete_optimizer"])
df["ratio"] = df["target_length"] / df["num_free_tokens"]
df["memorized"] = df["ratio"] > 1
print(df[["cfg_model_name", "cfg_dataset", "cfg_data_idx", "cfg_discrete_optimizer", "ratio", "memorized",
          "success"]].round(2).to_markdown())

# Make summary counting the average ratio and success rate for each dataset and discrete_optimizer include counts
summary = df.groupby(["cfg_model_name", "cfg_dataset", "cfg_discrete_optimizer"]).agg(
    {"ratio": "mean", "memorized": "mean", "success": "count"}).round(2)
print(summary.to_markdown())
print(f"dataframe shape: {df.shape}")
