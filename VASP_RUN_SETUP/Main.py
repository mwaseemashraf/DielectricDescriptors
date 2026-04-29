import os
import shutil
import numpy as np

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp import Poscar

from toolbox import (
    up, down, filechk, POTCAR_compiler, add_or_update_incar_tag,
    create_shell_directory_suffix, submit_sbatch_job, submit_sbatch_job_with_dependency
)


# CPU or GPU job script
job_script_selected = "job_script_cpu.sh"
potcar_functional="PBE"  # or "LDA" or LDA_US, passed to MPRelaxSet to determine POTCAR symbols
#Use the following for LDA_US
# if potcar_functional == "LDA_US":
#     user_potcar_settings = {
#         "Zr": "Zr",
#         "O": "O",
#     }
# else:
#     user_potcar_settings = {
#         "Zr": "Zr_sv",
#         "O": "O",
#     }
suffix = "4_node"
# -------------------------
# USER SETTINGS
# -------------------------
fakerun = 0  # set to 0 to actually submit jobs
kmesh = [3, 3, 3]

# Prefer setting MP API key via environment:
# export MP_API_KEY="..."
api_key = os.environ.get("MP_API_KEY", "2Kph2GG3x9ryu4tplcEN4VVKzdV4ilKc")

root_dir = os.getcwd()
poscar_folder = os.path.join(root_dir, "POSCARS_Enumeration")  # folder containing POSCAR_* files if using local POSCARs

# Choose exactly one:
#   "local" -> read POSCAR_* files from poscar_folder
#   "mp"    -> fetch structures from Materials Project using mp_ids
STRUCTURE_SOURCE = "local"  # "local" or "mp"

# If STRUCTURE_SOURCE == "mp", provide the mp-ids here:
mp_ids = ["mp-2574"]  # example mp-ids; replace with your own

# Required supporting files to copy into each batch directory
supporting_files = [
    'job_script_gpu.sh',
    'job_script_cpu.sh',
    'job_script_cpu_multinode.sh',
    'job_script_interm.sh',
    'toolbox.py',
    'SolveDielectric.py'
]


# -------------------------
# MATERIALS PROJECT FETCH
# -------------------------
def get_structure_from_materials_project(mp_id: str, api_key: str):
    """
    Fetch a Structure from Materials Project by mp-id.

    Tries the newer mp-api client first, then falls back to pymatgen's legacy MPRester.
    """
    # Newer client (mp-api)
    try:
        from mp_api.client import MPRester  # type: ignore
        with MPRester(api_key) as mpr:
            # This returns a pymatgen Structure
            return mpr.get_structure_by_material_id(mp_id)
    except Exception as e_new:
        # Legacy pymatgen MPRester
        try:
            from pymatgen.ext.matproj import MPRester  # type: ignore
            with MPRester(api_key) as mpr:
                return mpr.get_structure_by_material_id(mp_id)
        except Exception as e_old:
            raise RuntimeError(
                f"Failed to fetch structure for {mp_id}.\n"
                f"mp-api client error: {e_new}\n"
                f"legacy pymatgen MPRester error: {e_old}"
            )


# -------------------------
# CORE WORKFLOW
# -------------------------
def prepare_and_run(structure, structure_label: str, original_poscar_path: str | None = None):
    """
    Prepare and (optionally) submit a VASP relaxation + dielectric job.

    Parameters
    ----------
    structure : pymatgen Structure
    structure_label : str
        Used to name directories and set SYSTEM tag. Example: "POSCAR_Ce8" or "mp-2574"
    original_poscar_path : str | None
        If provided, the original POSCAR file will be copied into the batch directory.
        If None, a POSCAR_<structure_label> will be written into the batch directory.
    """
    print(f"\nProcessing {structure_label}...")

    # Create a shell directory with suffix

    shell_dir = create_shell_directory_suffix(suffix)

    # Ensure label doesn't contain path separators
    safe_label = structure_label.replace("/", "_").replace("\\", "_")

    batch_dir = os.path.join(shell_dir, safe_label)
    os.makedirs(batch_dir, exist_ok=True)

    # Copy supporting files
    for f in supporting_files:
        shutil.copy(os.path.join(root_dir, f), batch_dir)

    # Put a POSCAR into batch_dir
    down(batch_dir)
    if original_poscar_path is not None:
        # copy the provided POSCAR file into the batch directory
        poscar_basename = os.path.basename(original_poscar_path)
        shutil.copy(original_poscar_path, poscar_basename)
        poscar_filename_in_batch = poscar_basename
    else:
        # write a POSCAR_<label> in the batch directory
        poscar_filename_in_batch = f"POSCAR_{safe_label}"
        Poscar(structure).write_file(poscar_filename_in_batch)

    # --- Relaxation setup ---
    user_incar_settings = {
        'SYSTEM': f'{safe_label} relaxation',
        'ALGO':'Normal',
        'PREC':'Accurate',
        'ENCUT':'600',
        'EDIFF':'1e-6',
        'EDIFFG':'-0.01',
        'ISIF':'8',
        'NCORE':'12',
        'ISPIN':'1',
        'ISMEAR':'0',
        'LREAL':'False',
        # 'NELM':'100',
        # 'NSW':'100',
        'IBRION':'2',
        'SIGMA':'0.05',
        'GGA':'PS',
        'KPAR':'4',
}
    kpoints = Kpoints.monkhorst_automatic(kpts=kmesh)
    # kpoints = Kpoints.gamma_automatic(kpts=kmesh)
    # vis = MPRelaxSet(structure, force_gamma=True, user_kpoints_settings=kpoints,user_incar_settings=user_incar_settings,user_potcar_functional=potcar_functional,user_potcar_settings=user_potcar_settings)
    vis = MPRelaxSet(structure, force_gamma=True, user_kpoints_settings=kpoints,user_incar_settings=user_incar_settings)

    # Create Optimize directory
    os.makedirs("Optimize", exist_ok=True)
    for f in ['job_script_cpu.sh', 'job_script_cpu_multinode.sh', 'job_script_gpu.sh', 'job_script_interm.sh',job_script_selected]:
        shutil.copy(f, "Optimize")
    down("Optimize")

    # Write INCAR, KPOINTS, POSCAR
    for name, data in zip(["INCAR", "KPOINTS", "POSCAR","POTCAR"], [vis.incar, vis.kpoints, vis.poscar, vis.potcar]):
        with open(name, 'w') as fp:
            fp.write(str(data))

    # Create POTCAR
    # open("POTCAR", 'w').close().           Changes
    # POTCAR_compiler(vis.potcar_symbols).   Changes

    # Update INCAR tags
    # incar_tags = {
    #     'SYSTEM': f'{safe_label} relaxation',
    #     'PREC': 'Accurate',
    #     'ENCUT': '650',
    #     'EDIFF': '1e-8',
    #     'EDIFFG': '-0.001',
    #     'ISIF': '3',
    #     'ISMEAR': '0',
    #     'SIGMA': '0.05',
    #     'ISYM': '2',
    #     'NCORE': '4',
    #     'LREAL':'FALSE'
    # }
    # for tag, value in incar_tags.items():
    #     add_or_update_incar_tag('INCAR', tag, value)

    # Submit relaxation job
    if fakerun == 0:
        jobID = submit_sbatch_job(job_script_selected)
    else:
        jobID = None
        print(f"[FAKE RUN] Skipping relaxation submission for {safe_label}")

    up("Optimize")

    # --- Dielectric job prep ---
    dielectric_dir = "Dielectric"
    os.makedirs(dielectric_dir, exist_ok=True)
    for f in ["job_script_cpu_multinode.sh", "job_script_cpu.sh", "job_script_interm.sh", "toolbox.py", "SolveDielectric.py",job_script_selected]:
        shutil.copy(os.path.join(root_dir, f), dielectric_dir)

    down(dielectric_dir)

    # If you want to customize INCAR_d tags, do it here (currently commented in your original)

    if fakerun == 0 and jobID:
        submit_sbatch_job_with_dependency("job_script_interm.sh", jobID)
    else:
        print(f"[FAKE RUN] Skipping dielectric submission for {safe_label}")

    # Return to batch dir then root
    up(batch_dir)
    os.chdir(root_dir)
    print(f"✅ Completed setup for {safe_label} (POSCAR used: {poscar_filename_in_batch})")


def main():
    if STRUCTURE_SOURCE.lower() == "local":
        # Existing behavior: scan POSCARS folder for POSCAR_* files
        if not os.path.isdir(poscar_folder):
            print(f"❌ POSCAR folder not found: {poscar_folder}")
            return

        poscar_files = [
            os.path.join(poscar_folder, f)
            for f in os.listdir(poscar_folder)
            if f.startswith("POSCAR_s")
        ]

        if not poscar_files:
            print("❌ No POSCAR files found in 'POSCARS' folder.")
            return

        print(f"Found {len(poscar_files)} POSCAR files. Starting processing...\n")
        for poscar_path in poscar_files:
            poscar_name = os.path.basename(poscar_path)  # e.g., POSCAR_Ce8
            structure = Poscar.from_file(poscar_path).structure
            prepare_and_run(structure, structure_label=poscar_name, original_poscar_path=poscar_path)

    elif STRUCTURE_SOURCE.lower() == "mp":
        if not mp_ids:
            print("❌ STRUCTURE_SOURCE='mp' but mp_ids is empty. Add one or more mp-ids to mp_ids.")
            return

        if not api_key or api_key.strip() == "":
            print("❌ No Materials Project API key found. Set MP_API_KEY env var or set api_key in the script.")
            return

        print(f"Fetching {len(mp_ids)} structures from Materials Project. Starting processing...\n")
        for mp_id in mp_ids:
            structure = get_structure_from_materials_project(mp_id, api_key)
            # Use mp-id as label; POSCAR will be written into the batch directory
            prepare_and_run(structure, structure_label=mp_id, original_poscar_path=None)

    else:
        print(f"❌ Unknown STRUCTURE_SOURCE='{STRUCTURE_SOURCE}'. Use 'local' or 'mp'.")
        return


if __name__ == "__main__":
    main()
