import numpy as np
import shutil
import os
import time
import subprocess
from datetime import datetime
from itertools import product

import numpy as np

def get_material_data(material_id):
    # Define the matrices
    if material_id == "mp-2574":
        matrix = np.array([
            [3, 3, 3, 1, 0.62],
            [3, 3, 2, 1, 0.92],
            [3, 2, 2, 1, 1.39],            
            [3, 2, 2, 2, 2.78],
            [2, 2, 2, 2, 4.17],
            [3, 2, 1, 2, 5.56],
            [2, 2, 2, 3, 6.25],
            [2, 1, 1, 1, 8.33]
        ])
    elif material_id == "mp-2858":
        matrix = np.array([
            [3, 3, 3, 2, 0.62],
            [3, 3, 2, 2, 0.92],
            [3, 2, 1, 1, 1.39],            
            [3, 2, 1, 2, 2.78],
            [2, 2, 1, 2, 4.17],
            [3, 1, 1, 2, 5.56],
            [2, 2, 1, 3, 6.25],
            [1, 1, 1, 1, 8.33]
        ])
    else:
        raise ValueError(f"Unknown material ID: {material_id}")
    
    # Extract triplets, replace limits, and molpercents
    triplets = [tuple(row[:3].astype(int)) for row in matrix]
    replace_limits = [int(row[3]) for row in matrix]
    molpercent = [str(row[4]) for row in matrix]

    return triplets, replace_limits, molpercent




def kpoints_for_supercell(supercell_size, base_mesh=8):
    """
    Generate k-point mesh inversely proportional to supercell size.

    Args:
        supercell_size (list of int): e.g. [2, 2, 2]
        base_mesh (int): e.g. 8 (for 1x1x1 supercell)

    Returns:
        list of int: k-point mesh e.g. [4, 4, 4]
    """
    return [max(base_mesh // s, 1) for s in supercell_size]


def get_unique_supercell_triplets(min_val=1, max_val=4):
    """
    Generates unique [a, b, c] triplets in the range [min_val, max_val]
    such that only one triplet per unique volume (a * b * c) is included.

    Args:
        min_val (int): Minimum value for a, b, and c (inclusive).
        max_val (int): Maximum value for a, b, and c (inclusive).

    Returns:
        List of unique triplets [a, b, c] as lists.
    """
    seen_products = set()
    unique_triplets = []

    for a, b, c in product(range(min_val, max_val + 1), repeat=3):
        volume = a * b * c
        if volume not in seen_products:
            seen_products.add(volume)
            unique_triplets.append([a, b, c])

    return unique_triplets

def make_supercell_matrix(a, b, c):
    return [[a, 0, 0], [0, b, 0], [0, 0, c]]

def create_shell_directory(base_path='.', counter_file='batch_counter.txt'):
    """
    Creates a new shell directory with format: Batch_<batch#>_YYYYMMDD_HHMMSS.
    The batch number is auto-incremented and stored in a counter file.

    Args:
        base_path (str): The parent directory where the shell directory is created.
        counter_file (str): Path to the file storing the batch number counter.

    Returns:
        str: The full path of the created shell directory.
    """
    # Ensure counter file is in the base path
    counter_file = os.path.join(base_path, counter_file)

    # Read and update batch number
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            batch_number = int(f.read().strip()) + 1
    else:
        batch_number = 1

    with open(counter_file, 'w') as f:
        f.write(str(batch_number))

    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create directory name
    dir_name = f"Batch_GPU_{batch_number}_{timestamp}"
    full_path = os.path.join(base_path, dir_name)

    # Make the directory
    os.makedirs(full_path, exist_ok=True)

    return full_path

import os
from datetime import datetime

import os
from datetime import datetime

def create_shell_directory_suffix(suffix=''):
    """
    Creates a new shell directory with format: Batch_<batch#>_YYYYMMDD_HHMMSS[_suffix].
    The batch number is auto-incremented and stored in a counter file.

    Args:
        suffix (str): Optional string to append at the end of the directory name.

    Returns:
        str: The full path of the created shell directory.
    """
    base_path = '.'  # fixed base path
    counter_file = 'batch_counter.txt'  # fixed counter file name

    # Ensure counter file is in the base path
    counter_file_path = os.path.join(base_path, counter_file)

    # Read and update batch number
    if os.path.exists(counter_file_path):
        with open(counter_file_path, 'r') as f:
            batch_number = int(f.read().strip()) + 1
    else:
        batch_number = 1

    with open(counter_file_path, 'w') as f:
        f.write(str(batch_number))

    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build directory name with optional suffix
    if suffix:
        dir_name = f"Batch_{batch_number}_{timestamp}_{suffix}"
    else:
        dir_name = f"Batch_{batch_number}_{timestamp}"

    full_path = os.path.join(base_path, dir_name)

    # Make the directory
    os.makedirs(full_path, exist_ok=True)

    return full_path



def add_or_update_incar_tag(file_path, tag, new_value):
    """
    Add or update a tag in the VASP INCAR file.

    Parameters:
        - file_path (str): Path to the INCAR file.
        - tag (str): The tag to be added or updated.
        - new_value (str): The value for the tag.

    Returns:
	None
    """
    # Read the content of the INCAR file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tag_found = False

    # Update the value of the specified tag if it exists
    for i in range(len(lines)):
        if lines[i].strip().startswith(tag):
            lines[i] = f"{tag} = {new_value}\n"
            tag_found = True
            break

    # Add the tag with the specified value if it doesn't exist
    if not tag_found:
        lines.append(f"{tag} = {new_value}\n")

    # Write the updated content back to the INCAR file
    with open(file_path, 'w') as file:
        file.writelines(lines)



def POTCAR_compiler(potcar_symbols):
    #VASP_POTCAR_PATH="/fslhome/mwa32/fsl_groups/fslg_msg_vasp/potpaw_PBE/"
    # VASP_POTCAR_PATH="/home/mwa32/atomate_g/VASP/vasp.6.4.2_WSM2/potpaw_PBE/"
    VASP_POTCAR_PATH="/home/mwa32/atomate_g/VASP/vasp.6.4.2_WSM2/potUSPP_LDA/"
    Des="POTCAR"
    lis=[]
    for i in range(len(potcar_symbols)):
        #F=vis.potcar_symbols[i]
        lis=lis+[VASP_POTCAR_PATH+potcar_symbols[i]+"/"+"POTCAR"]
        with open(Des,"wb")as wfd:
            for files in lis:
                with open(files,"rb")as fd:
                    shutil.copyfileobj(fd,wfd)

def POSCAR_counter():
    POSCAR_num=0
    for i in range(999):
        if i < 10:
            cmd="POSCAR-00"+str(i)
        elif i>10 and i<100:
            cmd="POSCAR-0"+str(i)
        else:
            cmd="POSCAR-"+str(i)
        if os.path.exists(cmd):
            print("Found"+" "+cmd+" file")
            POSCAR_num=POSCAR_num+1
        else:
            i=999
    print(str(POSCAR_num)+" POSCARS found")
    return POSCAR_num



def up(file):
    absolutepath = os.path.abspath(file)
    fileDirectory = os.path.dirname(absolutepath)
    parentDirectory = os.path.dirname(fileDirectory)
    os.chdir(parentDirectory)

def down(file):
    os.chdir(file)

def filechk(file):
    if os.path.exists(file):
        while os.stat(file).st_size == 0:
            print("Let it fill")
            time.sleep(10)
            return
    else:
        print("Let's wait for CONTCAR here :(")
        time.sleep(10)
        filechk(file)

def submit_sbatch_job(script_path):
    try:
        # Run sbatch command to submit the job and capture the output
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True, check=True)

        # Extract the job ID from the command output
        job_id_str = result.stdout.strip().split()[-1]
        job_id = int(job_id_str)

        # Print the job ID
        print(f"Submitted job with ID: {job_id}")

        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting sbatch job: {e}")
        return None

# Replace 'your_script.sh' with the actual path to your sbatch script
#script_path = 'your_script.sh'
#job_id = submit_sbatch_job(script_path)

def submit_sbatch_job_with_dependency(script_path, dependency_job_id):
    try:
        # Run the sbatch command to submit a job with dependency
        subprocess.run(["sbatch", "--dependency=afterok:" + str(dependency_job_id), script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting sbatch job with dependency: {e}")
        
def parser():
    from pymatgen.io.vasp.outputs import Outcar, Vasprun
    vr = Vasprun('vasprun.xml')
    outcar = Outcar('OUTCAR')
    #print(vr.as_dict())
    print(outcar.as_dict())
    
    return None