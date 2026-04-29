import os
import shutil
from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.sets import MPStaticSet

from toolbox import up, down, submit_sbatch_job



# os.environ["PMG_VASP_PSP_DIR"] = (
#     "/home/mwa32/atomate_g/VASP/vasp.6.4.2_WSM2/potpaw_LDA"
# )
def safe_copy(src: str, dst: str) -> bool:
    """Copy src -> dst if it exists. Return True if copied."""
    if os.path.exists(src):
        shutil.copy(src, dst)
        return True
    return False


# -------------------------
# Workflow: Optimize -> Dielectric
# -------------------------
up("Dielectric")

# Required inputs from Optimize
safe_copy("Optimize/CONTCAR", "Dielectric/CONTCAR")
safe_copy("Optimize/POTCAR", "Dielectric/POTCAR")
safe_copy("Optimize/KPOINTS", "Dielectric/KPOINTS")


# Optional restart files (only if they exist)
wcar_copied = safe_copy("Optimize/WAVECAR", "Dielectric/WAVECAR")
safe_copy("Optimize/CHGCAR", "Dielectric/CHGCAR")

down("Dielectric")

# Use relaxed geometry as starting structure
if not os.path.exists("CONTCAR"):
    raise FileNotFoundError("Dielectric/CONTCAR not found. Ensure Optimize/CONTCAR was produced.")
shutil.copy("CONTCAR", "POSCAR")

structure = Poscar.from_file("POSCAR").structure

# System-agnostic MAGMOM: your intent was all zeros; match number of sites
magmom = {el.symbol: 0.0 for el in structure.composition.elements}

# If WAVECAR is missing, ISTART=1 will fail; fall back safely.
istart = 1 if os.path.exists("WAVECAR") else 0

# IMPORTANT: do NOT hard-code k-mesh here. Provide via env var or pre-existing KPOINTS.
# Approach:
# - If KPOINTS already exists in Dielectric (maybe you copied it), keep it.
# - Otherwise, let MPStaticSet generate a reasonable default.
use_existing_kpoints = os.path.exists("KPOINTS")

# Your requested tags
user_incar_settings = {
    "ALGO": "Normal",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "ENCUT": 600,
    "IBRION": 8,
    "ISMEAR": 0,
    "SIGMA":"0.05",
    "ISPIN": 1,
    "LEPSILON": True,
    "LREAL": "False",
    # "NELM": 100,
    "PREC": "Accurate",
    "NSW":1,
    "ADDGRID": True,
    "GGA": "PS",
    "NCORE": 12,
    "KPAR":4,
}

# Build set
vis = MPStaticSet(
    structure,
    force_gamma=True,
    user_incar_settings=user_incar_settings,
)

# Write INCAR
with open("INCAR", "w") as f:
    f.write(str(vis.incar))

# Write KPOINTS only if none exists already (avoid hard-coding k-mesh)
if not use_existing_kpoints:
    with open("KPOINTS", "w") as f:
        f.write(str(vis.kpoints))

# POTCAR should already be copied; MPStaticSet will not generate it unless asked.
if not os.path.exists("POTCAR"):
    raise FileNotFoundError("Dielectric/POTCAR not found. Copy it from Optimize.")

# Submit job
submit_sbatch_job("job_script_cpu.sh")

up("Dielectric")
