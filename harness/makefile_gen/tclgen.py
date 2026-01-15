
DEFAULT_CONFIG = {
    "device": "u55c",
    "frequency": 300,
    "mode": "csim|csynth|cosim|impl",
}

PART_NUMBER = {
    # Reference: https://github.com/Xilinx/XilinxBoardStore/tree/2022.2
    # Embedded
    "ultra96v2": "xczu3eg-sbva484-1-i",
    "pynqz2": "xc7z020clg400-1",
    "zedboard": "xc7z020clg484-1",
    # Zynq
    "zcu102": "xczu9eg-ffvb1156-2-e",
    "zcu104": "xczu7ev-ffvc1156-2-e",
    "zcu106": "xczu7ev-ffvc1156-2-e",
    "zcu111": "xczu28dr-ffvg1517-2-e",
    # Versal
    "vck190": "xcvc1902-vsva2197-2MP-e-S",
    "vhk158": "xcvh1582-vsva3697-2MP-e-S-es1",
    # Alveo
    # https://github.com/Xilinx/XilinxBoardStore/pull/434
    "u55c": "xcu55c-fsvh2892-2L-e",
    "u200": "xcu200-fsgd2104-2-e",
    "u250": "xcu250-figd2104-2L-e",
    "u280": "xcu280-fsvh2892-2L-e",
}

def codegen_tcl(top, configs):
    out_str = """
#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj hls.output

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

"""
    out_str += f'# Top function of the design is "{top}"\n'
    out_str += f"set_top {top}\n"
    out_str += """
# Add design and testbench files
add_files kernel.cpp
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"
"""
    device = configs["device"]
    frequency = configs["frequency"]
    mode = configs["mode"]
    if device not in PART_NUMBER:
        raise RuntimeError(
            f"Device {device} not supported. Available devices: {list(PART_NUMBER.keys())}"
        )
    out_str += f"\n# Target device is {device}\n"
    out_str += f"set_part {{{PART_NUMBER[device]}}}\n\n"
    out_str += "# Target frequency\n"
    out_str += f"create_clock -period {1000 / frequency:.2f}\n\n"
    out_str += "# Run HLS\n"
    if "csim" in mode or "sw_emu" in mode:
        out_str += "csim_design -O\n"
    if "csyn" in mode or "debug" in mode:
        out_str += "csynth_design\n"
    if "cosim" in mode or "hw_emu" in mode:
        out_str += "cosim_design\n"
    if "impl" in mode or "hw" in mode:
        if device in {"ultra96v2", "pynqz2", "zedboard"}:
            # Embedded boards: export IP only, bitstream happens in Python/Vivado later
            out_str += "export_design -rtl verilog -format ip_catalog\n"
        else:
            # Other platforms: run full impl in HLS
            out_str += "export_design -flow impl\n"
    out_str += "\nexit\n"
    return out_str
