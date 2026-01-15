
from makefile_gen.descgen import generate_description_file
from makefile_gen.makegen import generate_makefile
from makefile_gen.tclgen import DEFAULT_CONFIG, codegen_tcl
import os

project = "/home/dahaoming/hls-projs/gemm-axi4"
src_path = os.path.join(os.path.dirname(__file__), "makefile_gen/description.json")
dst_path = os.path.join(project, "description.json")
with open(f"{project}/run.tcl", "w", encoding="utf-8") as outfile:
  outfile.write(codegen_tcl("forward", DEFAULT_CONFIG))
    
generate_description_file("forward", src_path, dst_path, frequency=300)
generate_makefile(dst_path, project)
# print(dst_path)
# print(src_path)
