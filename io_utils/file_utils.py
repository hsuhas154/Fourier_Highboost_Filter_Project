# io/file_utils.py
"""
File naming, parameter recording, and zipping helpers.
"""

import os
import datetime
import zipfile
from typing import Dict

def make_result_filename(
    projname: str,
    input_path: str,
    filter_name: str,
    D0: float,
    r: float,
    channel: str,
    desc: str,
    ext: str = "png",
    outdir: str = ".",
) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_desc = str(desc).replace(" ", "_")
    fname = f"{projname}_{base}_{filter_name}_D0-{int(D0)}_r-{r}_{channel}_{safe_desc}_{timestamp}.{ext}"
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, fname)

def save_parameters_txt(outdir: str, params: Dict):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "parameters.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
    return path

def zip_results(dir_to_zip: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_to_zip):
            for file in files:
                full = os.path.join(root, file)
                arcname = os.path.relpath(full, start=dir_to_zip)
                zf.write(full, arcname)
    return zip_path
