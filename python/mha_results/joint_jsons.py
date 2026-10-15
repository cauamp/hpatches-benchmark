import glob
import json

import pandas as pd

# print as a table

files = glob.glob("*.json")


all_data = {}
for file in files:
    id = file.split(".")[0]  # Extract the ID from the filename

    with open(file, "r") as f:
        data = json.load(f)

    for k, v in data["illumination"].items():
        data[f"illumination_{k}"] = round(float(v), 3)
    for k, v in data["viewpoint"].items():
        data[f"viewpoint_{k}"] = round(float(v), 3)

    data.pop("illumination", None)  # Remove the original illumination dictionary
    data.pop("viewpoint", None)  # Remove the original viewpoint dictionary
    print(f"Processed file: {file}, ID: {id}, Data: {data}")

    all_data[id] = data

# One row per JSON file
df = pd.DataFrame.from_dict(all_data, orient="index")
df.index.name = "method"
# sort by method
df = df.sort_index()

# Save
df.to_csv("output.csv")
