
from sys import argv
import json

def generate_description_file(top, src_path, dst_path, frequency):
    with open(src_path, "r", encoding="utf-8") as f:
        desc = f.read()
    desc = desc.replace("top", top)
    desc = json.loads(desc)
    desc["containers"][0]["ldclflags"] += f"  --kernel_frequency {frequency}"
    with open(dst_path, "w", encoding="utf-8") as outfile:
        json.dump(desc, outfile, indent=2)
