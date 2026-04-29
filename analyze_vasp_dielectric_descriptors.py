import csv
import os
import re
import shutil
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(__file__)
VASP_DATA = os.path.join(ROOT, "VASP_DATA")
OPTIMIZED_DIR = os.path.join(ROOT, "optimized_structures")
RESULTS_DIR = os.path.join(ROOT, "vasp_descriptor_results")


def numeric_poscar_key(name):
    match = re.search(r"POSCAR\.(\d+)$", name)
    if match:
        return int(match.group(1))
    return name


def find_structure_runs():
    runs = []
    for batch in os.listdir(VASP_DATA):
        batch_path = os.path.join(VASP_DATA, batch)
        if not os.path.isdir(batch_path):
            continue
        for name in os.listdir(batch_path):
            if not name.startswith("POSCAR."):
                continue
            structure_path = os.path.join(batch_path, name)
            if os.path.isdir(structure_path):
                runs.append(
                    {
                        "structure": name,
                        "batch": batch,
                        "path": structure_path,
                        "optimize": os.path.join(structure_path, "Optimize"),
                        "dielectric": os.path.join(structure_path, "Dielectric"),
                    }
                )
    return sorted(runs, key=lambda r: numeric_poscar_key(r["structure"]))


def copy_optimized_contcars(runs):
    os.makedirs(OPTIMIZED_DIR, exist_ok=True)
    copied = []
    for run in runs:
        source = os.path.join(run["optimize"], "CONTCAR")
        target = os.path.join(OPTIMIZED_DIR, run["structure"])
        if not os.path.exists(source):
            raise FileNotFoundError(f"Missing optimized CONTCAR: {source}")
        shutil.copyfile(source, target)
        copied.append(target)
    return copied


def parse_poscar(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = [line.rstrip() for line in fh if line.strip()]
    if len(raw_lines) < 9:
        raise ValueError(f"POSCAR file too short: {path}")

    scale = float(raw_lines[1].split()[0])
    lattice = np.array([list(map(float, raw_lines[i].split()[:3])) for i in range(2, 5)]) * scale
    species = raw_lines[5].split()
    counts = list(map(int, raw_lines[6].split()))

    coord_line_index = 7
    if raw_lines[coord_line_index].lower().startswith("s"):
        coord_line_index += 1
    coord_type = raw_lines[coord_line_index].lower()
    coord_start = coord_line_index + 1
    coords = np.array(
        [list(map(float, raw_lines[i].split()[:3])) for i in range(coord_start, coord_start + sum(counts))]
    )

    if coord_type.startswith("cart") or coord_type.startswith("k"):
        cart_coords = coords * scale
    else:
        cart_coords = coords.dot(lattice)

    species_list = []
    for sp, count in zip(species, counts):
        species_list.extend([sp] * count)

    return lattice, species_list, cart_coords


def min_image_vector(dvec, inv_latt, lattice):
    frac = dvec.dot(inv_latt)
    frac -= np.round(frac)
    return frac.dot(lattice.T)


def min_image_distance_matrix(coords_a, coords_b, lattice, inv_latt):
    a = np.asarray(coords_a)
    b = np.asarray(coords_b)
    dists = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        diff = a[i] - b
        frac = diff.dot(inv_latt)
        frac -= np.round(frac)
        image_diff = frac.dot(lattice.T)
        dists[i] = np.linalg.norm(image_diff, axis=1)
    return dists


def neighbor_environment(center, other_positions, inv_latt, lattice, max_neighbors=12):
    diffs = other_positions - center
    vectors = np.array([min_image_vector(d, inv_latt, lattice) for d in diffs])
    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances)
    distances = distances[order]
    vectors = vectors[order]
    if len(distances) == 0:
        return np.array([]), np.array([])

    gaps = np.diff(distances)
    if len(gaps) > 0:
        gap_index = np.argmax(gaps[:max_neighbors])
        cutoff = distances[gap_index] + 0.5 * gaps[gap_index]
        selected = np.where(distances <= cutoff)[0]
    else:
        selected = np.arange(min(max_neighbors, len(distances)))
    selected = selected[:max_neighbors]
    return distances[selected], vectors[selected]


def compute_site_symmetry(neighbor_vectors):
    if len(neighbor_vectors) < 2:
        return np.nan, np.nan, np.nan

    distances = np.linalg.norm(neighbor_vectors, axis=1)
    dist_var = np.std(distances) / np.mean(distances)
    normed = neighbor_vectors / np.maximum(distances[:, None], 1e-8)

    inertia = normed.T @ normed
    eigvals = np.linalg.eigvalsh(inertia)
    shape_anisotropy = np.std(eigvals) / np.maximum(np.mean(eigvals), 1e-8)

    angles = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            cosang = np.clip(np.dot(normed[i], normed[j]), -1.0, 1.0)
            angles.append(np.arccos(cosang))
    angles = np.array(angles)
    angle_mean = np.degrees(np.mean(angles)) if len(angles) else np.nan
    angle_std = np.degrees(np.std(angles)) if len(angles) else np.nan
    score = 1.0 / (1.0 + dist_var + shape_anisotropy + angle_std / 180.0)
    return score, angle_mean, angle_std


def cluster_geometry_label(positions, inv_latt, lattice, order):
    distances = []
    for i, j in combinations(range(order), 2):
        d = np.linalg.norm(min_image_vector(positions[j] - positions[i], inv_latt, lattice))
        distances.append(round(d, 5))
    return tuple(sorted(distances))


def standardize_label(label):
    return ",".join(f"{x:.3f}" for x in label)


def compute_descriptors(poscar_path):
    lattice, species_list, cart_coords = parse_poscar(poscar_path)
    inv_latt = np.linalg.inv(lattice.T)

    ce_indices = [i for i, sp in enumerate(species_list) if sp.lower().startswith("ce")]
    zr_indices = [i for i, sp in enumerate(species_list) if sp.lower().startswith("zr")]
    o_indices = [i for i, sp in enumerate(species_list) if sp.lower().startswith("o")]
    dopant_indices = [i for i, sp in enumerate(species_list) if sp.lower() in {"ce", "zr"}]

    if not ce_indices:
        raise RuntimeError(f"No Ce atoms found in {poscar_path}")
    if not o_indices:
        raise RuntimeError(f"No O atoms found in {poscar_path}")

    ce_coords = cart_coords[ce_indices]
    if len(ce_coords) >= 2:
        ce_dists = min_image_distance_matrix(ce_coords, ce_coords, lattice, inv_latt)
        ce_ce_min = float(np.min(ce_dists[np.triu_indices(len(ce_coords), k=1)]))
    else:
        translations = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if (i, j, k) != (0, 0, 0):
                        translations.append(np.linalg.norm(i * lattice[0] + j * lattice[1] + k * lattice[2]))
        ce_ce_min = float(np.min(translations))

    ce_zr_min = float(np.min(min_image_distance_matrix(ce_coords, cart_coords[zr_indices], lattice, inv_latt)))
    ce_o_min = float(np.min(min_image_distance_matrix(ce_coords, cart_coords[o_indices], lattice, inv_latt)))

    ce_site_rows = []
    for ce_pos in ce_coords:
        o_distances, o_vectors = neighbor_environment(ce_pos, cart_coords[o_indices], inv_latt, lattice)
        symmetry_score, angle_mean_deg, angle_std_deg = compute_site_symmetry(o_vectors)
        ce_site_rows.append(
            {
                "o_neighbors": len(o_vectors),
                "o_mean_dist": float(np.mean(o_distances)) if len(o_distances) else np.nan,
                "o_dist_std": float(np.std(o_distances)) if len(o_distances) else np.nan,
                "symmetry_score": float(symmetry_score),
                "angle_mean_deg": float(angle_mean_deg),
                "angle_std_deg": float(angle_std_deg),
                "angle_descriptor": float(1.0 / (1.0 + angle_std_deg / 20.0)),
            }
        )

    descriptors = {
        "n_ce": len(ce_indices),
        "ce_ce_min": ce_ce_min,
        "ce_zr_min": ce_zr_min,
        "ce_o_min": ce_o_min,
    }
    descriptors.update(compute_lattice_descriptors(lattice, species_list, cart_coords))
    descriptors.update(compute_neighbor_variation_descriptors(lattice, species_list, cart_coords))
    for key in ce_site_rows[0]:
        descriptors[key] = float(np.nanmean([row[key] for row in ce_site_rows]))

    dop_positions = cart_coords[dopant_indices]
    sigmas = np.array([1 if species_list[i].lower() == "ce" else -1 for i in dopant_indices])
    cluster_values = {}
    for order in (2, 3, 4):
        by_label = {}
        for cluster in combinations(range(len(dop_positions)), order):
            label = cluster_geometry_label(dop_positions[list(cluster)], inv_latt, lattice, order)
            corr = float(np.prod(sigmas[list(cluster)]))
            by_label.setdefault(label, []).append(corr)
        for label, values in by_label.items():
            cluster_values[f"cluster_o{order}_{standardize_label(label)}"] = float(np.mean(values))

    return descriptors, cluster_values


def compute_lattice_descriptors(lattice, species_list, cart_coords):
    lengths = np.linalg.norm(lattice, axis=1)
    volume = float(abs(np.linalg.det(lattice)))
    return {
        "lattice_a": float(lengths[0]),
        "lattice_b": float(lengths[1]),
        "lattice_c": float(lengths[2]),
        "volume": volume,
        "volume_per_atom": volume / len(species_list),
        "c_over_mean_ab": float(lengths[2] / np.mean(lengths[:2])),
        "lattice_length_cv": float(np.std(lengths) / np.mean(lengths)),
    }


def compute_neighbor_variation_descriptors(lattice, species_list, cart_coords, nearest_n=8):
    inv_latt = np.linalg.inv(lattice.T)
    all_cv = []
    ce_o_cv = []
    ce_o_max = []
    for i, pos in enumerate(cart_coords):
        other_indices = [j for j in range(len(cart_coords)) if j != i]
        distances = min_image_distance_matrix([pos], cart_coords[other_indices], lattice, inv_latt)[0]
        selected = np.sort(distances)[: min(nearest_n, len(distances))]
        if len(selected) > 1 and np.mean(selected) > 0:
            all_cv.append(np.std(selected) / np.mean(selected))

        if species_list[i].lower().startswith("ce"):
            o_indices = [j for j, sp in enumerate(species_list) if sp.lower().startswith("o")]
            o_distances = min_image_distance_matrix([pos], cart_coords[o_indices], lattice, inv_latt)[0]
            selected_o = np.sort(o_distances)[: min(nearest_n, len(o_distances))]
            if len(selected_o) > 1 and np.mean(selected_o) > 0:
                ce_o_cv.append(np.std(selected_o) / np.mean(selected_o))
                ce_o_max.append(np.max(selected_o))

    return {
        "mean_neighbor_distance_variation": float(np.mean(all_cv)) if all_cv else np.nan,
        "max_neighbor_distance_variation": float(np.max(all_cv)) if all_cv else np.nan,
        "ce_o_distance_variation": float(np.mean(ce_o_cv)) if ce_o_cv else np.nan,
        "ce_o_max_distance": float(np.mean(ce_o_max)) if ce_o_max else np.nan,
    }


def compute_relaxation_descriptors(initial_poscar_path, optimized_poscar_path):
    initial_lattice, initial_species, initial_cart = parse_poscar(initial_poscar_path)
    opt_lattice, opt_species, opt_cart = parse_poscar(optimized_poscar_path)
    if initial_species != opt_species or len(initial_cart) != len(opt_cart):
        raise RuntimeError(f"Initial and optimized POSCAR atom lists differ for {optimized_poscar_path}")

    inv_initial = np.linalg.inv(initial_lattice.T)
    displacements = np.array(
        [
            np.linalg.norm(min_image_vector(opt_cart[i] - initial_cart[i], inv_initial, initial_lattice))
            for i in range(len(initial_cart))
        ]
    )
    rows = {
        "relax_mean_disp": float(np.mean(displacements)),
        "relax_rms_disp": float(np.sqrt(np.mean(displacements**2))),
        "relax_max_disp": float(np.max(displacements)),
        "relax_volume_strain": float((abs(np.linalg.det(opt_lattice)) - abs(np.linalg.det(initial_lattice))) / abs(np.linalg.det(initial_lattice))),
    }
    for species_name in ("Ce", "Zr", "O"):
        indices = [i for i, sp in enumerate(initial_species) if sp.lower().startswith(species_name.lower())]
        if indices:
            rows[f"relax_{species_name.lower()}_mean_disp"] = float(np.mean(displacements[indices]))
            rows[f"relax_{species_name.lower()}_max_disp"] = float(np.max(displacements[indices]))
        else:
            rows[f"relax_{species_name.lower()}_mean_disp"] = np.nan
            rows[f"relax_{species_name.lower()}_max_disp"] = np.nan
    return rows


def parse_outcar(outcar_path):
    with open(outcar_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    tensor = None
    for i, line in enumerate(lines):
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION" in line:
            rows = []
            for j in range(i + 1, min(i + 8, len(lines))):
                nums = re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", lines[j])
                if len(nums) == 3:
                    rows.append([float(x) for x in nums])
                if len(rows) == 3:
                    break
            tensor = np.array(rows, dtype=float)
            break
    if tensor is None or tensor.shape != (3, 3):
        raise RuntimeError(f"Could not parse ionic dielectric tensor from {outcar_path}")

    cpu_time = np.nan
    for line in lines:
        if "Total CPU time used" in line:
            cpu_time = float(re.findall(r"[-+]?\d+\.\d+", line)[0])
            break

    modes = []
    mode_pattern = re.compile(
        r"^\s*(\d+)\s+(f(?:/i)?\s*=)\s*([-+]?\d+\.\d+)\s+THz\s+[-+]?\d+\.\d+\s+2PiTHz\s+([-+]?\d+\.\d+)\s+cm-1"
    )
    for line in lines:
        match = mode_pattern.match(line)
        if not match:
            continue
        sign = -1.0 if match.group(2).replace(" ", "") == "f/i=" else 1.0
        modes.append(
            {
                "mode": int(match.group(1)),
                "freq_thz": sign * float(match.group(3)),
                "freq_cm1": sign * float(match.group(4)),
            }
        )
    if len(modes) < 3:
        raise RuntimeError(f"Could not parse at least three phonon modes from {outcar_path}")

    lowest = sorted(modes, key=lambda row: row["freq_thz"])[:3]
    return tensor, cpu_time, lowest, modes


def compute_phonon_descriptors(modes):
    if len(modes) <= 3:
        raise RuntimeError("Need more than three phonon modes to define optical-mode descriptors")
    optical_modes = sorted(modes, key=lambda row: row["mode"])[:-3]
    signed_freqs = np.array([row["freq_thz"] for row in optical_modes], dtype=float)
    abs_freqs = np.abs(signed_freqs)
    low3 = np.sort(abs_freqs)[:3]
    softest = float(np.min(abs_freqs))
    floor = max(softest, 0.05)
    return {
        "optical_softest_abs_freq_thz": softest,
        "optical_softest_signed_freq_thz": float(signed_freqs[np.argmin(abs_freqs)]),
        "optical_softness_inv_freq2": float(1.0 / floor**2),
        "optical_lowest3_mean_abs_freq_thz": float(np.mean(low3)),
        "optical_low_mode_count_lt_2thz": int(np.sum(abs_freqs < 2.0)),
        "optical_low_mode_count_lt_3thz": int(np.sum(abs_freqs < 3.0)),
        "optical_imaginary_mode_count": int(np.sum(signed_freqs < 0.0)),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def ridge_predict_loocv(x, y, alpha=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    predictions = np.zeros(len(y))
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        x_train = x[train]
        y_train = y[train]
        mean = np.zeros(x_train.shape[1])
        std = np.ones(x_train.shape[1])
        for col in range(x_train.shape[1]):
            finite = np.isfinite(x_train[:, col])
            if np.any(finite):
                mean[col] = np.mean(x_train[finite, col])
                col_std = np.std(x_train[finite, col])
                std[col] = col_std if col_std > 0 else 1.0
        std[std == 0] = 1.0
        x_train = np.nan_to_num((x_train - mean) / std)
        x_test = np.nan_to_num((x[[held_out]] - mean) / std)

        design = np.column_stack([np.ones(len(x_train)), x_train])
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
        predictions[held_out] = float((np.column_stack([np.ones(1), x_test]) @ coef)[0])
    return predictions


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2


def plot_descriptor_scatter(rows, descriptor_names, target_name, output_path):
    ncols = 3
    nrows = int(np.ceil(len(descriptor_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    y = np.array([row[target_name] for row in rows], dtype=float)
    labels = [row["structure"] for row in rows]

    for ax, descriptor in zip(axes.ravel(), descriptor_names):
        x = np.array([row[descriptor] for row in rows], dtype=float)
        r = pearson_r(x, y)
        ax.scatter(x, y, s=42, edgecolors="black", linewidths=0.5)
        if np.isfinite(r):
            ax.text(0.04, 0.92, f"r = {r:.2f}", transform=ax.transAxes)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label.replace("POSCAR.", ""), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(descriptor)
        ax.set_ylabel("trace(eps_ionic)/3")
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(descriptor_names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_parity(rows, feature_groups, target_name, output_path):
    ncols = 3
    nrows = int(np.ceil(len(feature_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    y = np.array([row[target_name] for row in rows], dtype=float)
    lo = min(y)
    hi = max(y)
    pad = 0.05 * (hi - lo)

    summary = []
    for ax, (name, features) in zip(axes.ravel(), feature_groups.items()):
        x = np.array([[row.get(feature, np.nan) for feature in features] for row in rows], dtype=float)
        pred = ridge_predict_loocv(x, y)
        rmse, mae, r2 = regression_metrics(y, pred)
        summary.append({"descriptor_family": name, "n_features": len(features), "loocv_rmse": rmse, "loocv_mae": mae, "loocv_r2": r2})

        ax.scatter(y, pred, s=42, edgecolors="black", linewidths=0.5)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("DFT trace(eps_ionic)/3")
        ax.set_ylabel("LOOCV prediction")
        ax.set_title(name)
        ax.text(0.04, 0.88, f"RMSE = {rmse:.2f}\nR2 = {r2:.2f}", transform=ax.transAxes)
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(feature_groups) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return summary


def rank_single_descriptors(rows, descriptor_names, target_name):
    y = np.array([row[target_name] for row in rows], dtype=float)
    ranked = []
    for descriptor in descriptor_names:
        x = np.array([[row.get(descriptor, np.nan)] for row in rows], dtype=float)
        if np.sum(np.isfinite(x[:, 0])) < 3:
            continue
        pred = ridge_predict_loocv(x, y)
        rmse, mae, r2 = regression_metrics(y, pred)
        ranked.append(
            {
                "descriptor": descriptor,
                "pearson_r": pearson_r(x[:, 0], y),
                "loocv_rmse": rmse,
                "loocv_mae": mae,
                "loocv_r2": r2,
            }
        )
    return sorted(ranked, key=lambda row: row["loocv_rmse"])


def main():
    if not os.path.isdir(VASP_DATA):
        raise FileNotFoundError(f"VASP data folder not found: {VASP_DATA}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs = find_structure_runs()
    if not runs:
        raise RuntimeError("No POSCAR.* VASP runs found under VASP_DATA")

    copy_optimized_contcars(runs)

    result_rows = []
    descriptor_rows = []
    cluster_by_structure = {}

    for run in runs:
        outcar = os.path.join(run["dielectric"], "OUTCAR")
        optimized_poscar = os.path.join(OPTIMIZED_DIR, run["structure"])
        initial_poscar = os.path.join(run["path"], run["structure"])
        tensor, cpu_time, lowest_modes, all_modes = parse_outcar(outcar)
        descriptors, clusters = compute_descriptors(optimized_poscar)
        descriptors.update(compute_relaxation_descriptors(initial_poscar, optimized_poscar))
        descriptors.update(compute_phonon_descriptors(all_modes))

        eps_avg = float(np.trace(tensor) / 3.0)
        eigvals = np.linalg.eigvalsh(tensor)
        eps_anisotropy = float(np.max(eigvals) - np.min(eigvals))

        row = {
            "structure": run["structure"],
            "batch": run["batch"],
            "eps_xx": tensor[0, 0],
            "eps_yy": tensor[1, 1],
            "eps_zz": tensor[2, 2],
            "eps_xy": tensor[0, 1],
            "eps_yz": tensor[1, 2],
            "eps_zx": tensor[2, 0],
            "eps_trace_avg": eps_avg,
            "eps_anisotropy": eps_anisotropy,
            "total_cpu_time_sec": cpu_time,
        }
        for index, mode in enumerate(lowest_modes, start=1):
            row[f"lowest_mode_{index}"] = mode["mode"]
            row[f"lowest_freq_{index}_thz"] = mode["freq_thz"]
            row[f"lowest_freq_{index}_cm1"] = mode["freq_cm1"]
        result_rows.append(row)

        descriptor_row = {"structure": run["structure"], "eps_trace_avg": eps_avg, "eps_anisotropy": eps_anisotropy}
        descriptor_row.update(descriptors)
        descriptor_row.update(clusters)
        descriptor_rows.append(descriptor_row)
        cluster_by_structure[run["structure"]] = clusters

    result_fields = list(result_rows[0].keys())
    write_csv(os.path.join(RESULTS_DIR, "vasp_dielectric_summary.csv"), result_rows, result_fields)

    all_descriptor_fields = sorted({key for row in descriptor_rows for key in row.keys()})
    leading = ["structure", "eps_trace_avg", "eps_anisotropy"]
    descriptor_fields = leading + [field for field in all_descriptor_fields if field not in leading]
    write_csv(os.path.join(RESULTS_DIR, "descriptor_table.csv"), descriptor_rows, descriptor_fields)

    scalar_descriptors = [
        "ce_ce_min",
        "ce_zr_min",
        "ce_o_min",
        "o_neighbors",
        "o_mean_dist",
        "o_dist_std",
        "symmetry_score",
        "angle_std_deg",
        "angle_descriptor",
    ]
    physics_descriptors = [
        "optical_softest_abs_freq_thz",
        "optical_softest_signed_freq_thz",
        "optical_softness_inv_freq2",
        "optical_lowest3_mean_abs_freq_thz",
        "optical_low_mode_count_lt_2thz",
        "volume_per_atom",
        "c_over_mean_ab",
        "lattice_length_cv",
        "mean_neighbor_distance_variation",
        "max_neighbor_distance_variation",
        "ce_o_distance_variation",
        "ce_o_max_distance",
        "relax_mean_disp",
        "relax_rms_disp",
        "relax_max_disp",
        "relax_o_mean_disp",
        "relax_volume_strain",
    ]
    plot_descriptor_scatter(
        descriptor_rows,
        scalar_descriptors,
        "eps_trace_avg",
        os.path.join(RESULTS_DIR, "descriptor_property_scatter.png"),
    )
    plot_descriptor_scatter(
        descriptor_rows,
        physics_descriptors,
        "eps_trace_avg",
        os.path.join(RESULTS_DIR, "physics_descriptor_property_scatter.png"),
    )

    cluster_order_features = {
        order: sorted([field for field in descriptor_fields if field.startswith(f"cluster_o{order}_")])
        for order in (2, 3, 4)
    }
    feature_groups = {
        "Ce distance descriptors": ["ce_ce_min", "ce_zr_min", "ce_o_min"],
        "Ce-O local descriptors": ["o_neighbors", "o_mean_dist", "o_dist_std", "symmetry_score", "angle_std_deg", "angle_descriptor"],
        "Soft-mode descriptors": [
            "optical_softest_abs_freq_thz",
            "optical_softest_signed_freq_thz",
            "optical_softness_inv_freq2",
            "optical_lowest3_mean_abs_freq_thz",
            "optical_low_mode_count_lt_2thz",
            "optical_low_mode_count_lt_3thz",
            "optical_imaginary_mode_count",
        ],
        "Lattice/distortion descriptors": [
            "volume_per_atom",
            "c_over_mean_ab",
            "lattice_length_cv",
            "mean_neighbor_distance_variation",
            "max_neighbor_distance_variation",
            "ce_o_distance_variation",
            "ce_o_max_distance",
        ],
        "Relaxation descriptors": [
            "relax_mean_disp",
            "relax_rms_disp",
            "relax_max_disp",
            "relax_ce_mean_disp",
            "relax_zr_mean_disp",
            "relax_o_mean_disp",
            "relax_volume_strain",
        ],
        "Physics combined": [
            "optical_softest_abs_freq_thz",
            "optical_softness_inv_freq2",
            "optical_lowest3_mean_abs_freq_thz",
            "volume_per_atom",
            "c_over_mean_ab",
            "mean_neighbor_distance_variation",
            "ce_o_distance_variation",
            "relax_rms_disp",
            "relax_o_mean_disp",
            "relax_volume_strain",
        ],
        "Ce/Zr pair correlations": cluster_order_features[2],
        "Ce/Zr triplet correlations": cluster_order_features[3],
        "Ce/Zr quadruplet correlations": cluster_order_features[4],
    }
    summary = plot_parity(
        descriptor_rows,
        feature_groups,
        "eps_trace_avg",
        os.path.join(RESULTS_DIR, "descriptor_family_parity.png"),
    )
    candidate_single_descriptors = scalar_descriptors + physics_descriptors
    single_summary = rank_single_descriptors(descriptor_rows, candidate_single_descriptors, "eps_trace_avg")
    top_single_groups = {
        row["descriptor"]: [row["descriptor"]]
        for row in single_summary[:6]
    }
    plot_parity(
        descriptor_rows,
        top_single_groups,
        "eps_trace_avg",
        os.path.join(RESULTS_DIR, "top_single_descriptor_parity.png"),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "descriptor_performance_summary.csv"),
        summary,
        ["descriptor_family", "n_features", "loocv_rmse", "loocv_mae", "loocv_r2"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "single_descriptor_performance.csv"),
        single_summary,
        ["descriptor", "pearson_r", "loocv_rmse", "loocv_mae", "loocv_r2"],
    )

    notes_path = os.path.join(RESULTS_DIR, "plot_literature_notes.md")
    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Plot choices\n\n"
            "- `descriptor_property_scatter.png`: descriptor-property scatter plots, following the local-structure/order-parameter descriptor literature style for assessing whether a descriptor separates or trends with a target property.\n"
            "- `physics_descriptor_property_scatter.png`: the same descriptor-property scatter format for soft-mode, lattice, neighbor-variation, and relaxation descriptors motivated by dielectric-response literature.\n"
            "- `descriptor_family_parity.png`: calculated-vs-predicted parity plots, following cluster-expansion validation practice for testing descriptor families against DFT targets.\n"
            "- The target used for performance is `trace(eps_ionic)/3`, the orientational average of the parsed ionic dielectric tensor. The full tensor is preserved in `vasp_dielectric_summary.csv`.\n"
            "- The added soft-mode descriptors follow the DFPT relation in which ionic susceptibility is controlled by mode effective charges divided by phonon frequency squared. The added distortion descriptors follow oxide-permittivity ML studies that identify local geometric asymmetry and neighbor-distance variation as important features.\n"
        )

    print(f"Copied {len(runs)} optimized CONTCAR files to {OPTIMIZED_DIR}")
    print(f"Wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
