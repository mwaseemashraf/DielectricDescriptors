import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

ROOT = os.path.join(os.path.dirname(__file__), "training_set")
if not os.path.isdir(ROOT):
    raise FileNotFoundError(f"Dataset folder not found: {ROOT}")


def parse_poscar(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 9:
        raise ValueError(f"POSCAR file too short: {path}")

    scale_line = lines[1].split()
    if len(scale_line) == 1:
        scale = float(scale_line[0])
        lattice = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale
        species = lines[5].split()
        counts = list(map(int, lines[6].split()))
        coord_type = lines[7].lower()
        coords = np.array([list(map(float, lines[i].split())) for i in range(8, 8 + sum(counts))])
    else:
        scale = 1.0
        lattice = np.array([list(map(float, lines[i].split())) for i in range(1, 4)]) * scale
        species = lines[4].split()
        counts = list(map(int, lines[5].split()))
        coord_type = lines[6].lower()
        coords = np.array([list(map(float, lines[i].split())) for i in range(7, 7 + sum(counts))])

    if coord_type.startswith("cart"):
        cart_coords = coords
    else:
        cart_coords = coords.dot(lattice)

    species_list = []
    for sp, count in zip(species, counts):
        species_list.extend([sp] * count)

    return lattice, species_list, cart_coords


def min_image_dist(vec, inv_latt, lattice):
    frac = vec.dot(inv_latt)
    frac -= np.round(frac)
    return np.linalg.norm(frac.dot(lattice.T))


def cluster_geometry_label(positions, inv_latt, lattice, order):
    distances = []
    for i, j in combinations(range(order), 2):
        d = min_image_dist(positions[j] - positions[i], inv_latt, lattice)
        distances.append(round(d, 5))
    return tuple(sorted(distances))


def cluster_corr_value(sigmas):
    return np.prod(sigmas)


def compute_structure_correlations(lattice, positions, sigmas, orders=(2, 3, 4)):
    inv_latt = np.linalg.inv(lattice.T)
    structure_correlations = {order: {} for order in orders}
    for order in orders:
        for cluster in combinations(range(len(positions)), order):
            cluster_positions = positions[list(cluster)]
            label = cluster_geometry_label(cluster_positions, inv_latt, lattice, order)
            corr = cluster_corr_value([sigmas[i] for i in cluster])
            structure_correlations[order].setdefault(label, []).append(corr)
    for order in orders:
        for label, values in list(structure_correlations[order].items()):
            structure_correlations[order][label] = np.mean(values)
    return structure_correlations


def standardize_label(label):
    if len(label) == 1:
        return f"{label[0]:.3f}"
    return ",".join(f"{d:.3f}" for d in label)


def build_heat_matrix(all_structures, order, top_n=None):
    label_counts = {}
    for struct in all_structures:
        for label in struct[order].keys():
            label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts, key=lambda l: (-label_counts[l], l))
    if top_n is not None:
        sorted_labels = sorted_labels[:top_n]

    labels = sorted_labels
    matrix = np.zeros((len(all_structures), len(labels)))
    matrix.fill(np.nan)
    for si, struct in enumerate(all_structures):
        for ci, label in enumerate(labels):
            matrix[si, ci] = struct[order].get(label, np.nan)
    return labels, matrix


def plot_heatmap(matrix, row_labels, col_labels, title, output_path, vmax=1.0, vmin=-1.0):
    plt.figure(figsize=(max(10, len(col_labels) * 0.3), max(6, len(row_labels) * 0.6)))
    im = plt.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Cluster correlation")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks(range(len(col_labels)), [standardize_label(l) for l in col_labels], rotation=90, fontsize=8)
    plt.title(title)
    plt.xlabel("Cluster type label")
    plt.ylabel("Structure")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    files = sorted(
        [f for f in os.listdir(ROOT) if f.lower().startswith("poscar") or f.lower().endswith("vasp")],
        key=lambda x: int("".join(filter(str.isdigit, x))) if any(ch.isdigit() for ch in x) else x,
    )

    structures = []
    names = []
    for filename in files:
        lattice, species_list, cart_coords = parse_poscar(os.path.join(ROOT, filename))
        doping_indices = [i for i, sp in enumerate(species_list) if sp.lower() in {"ce", "zr"}]
        if not doping_indices:
            continue
        dop_positions = cart_coords[doping_indices]
        sigmas = np.array([1 if species_list[i].lower() == "ce" else -1 for i in doping_indices])
        structures.append(compute_structure_correlations(lattice, dop_positions, sigmas, orders=(2, 3, 4)))
        names.append(filename)

    if not structures:
        raise RuntimeError("No doping structures found")

    # Heatmaps for literature-style structure-by-cluster matrices.
    for order, top_n in [(2, None), (3, 15), (4, 12)]:
        labels, matrix = build_heat_matrix(structures, order, top_n=top_n)
        order_str = f"order_{order}"
        file_name = f"ce_zr_cluster_heatmap_{order_str}.png"
        title = (
            f"Ce/Zr cluster correlations (order {order})"
            + (" — top {} cluster types".format(top_n) if top_n else "")
        )
        plot_heatmap(matrix, names, labels, title, os.path.join(os.path.dirname(__file__), file_name))
        print(f"Saved heatmap: {file_name}")

    # Also save a summary of top cluster types and their frequencies
    summary_path = os.path.join(os.path.dirname(__file__), "ce_zr_cluster_heatmap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for order in (2, 3, 4):
            f.write(f"Order {order} top cluster types:\n")
            label_counts = {}
            for struct in structures:
                for label in struct[order].keys():
                    label_counts[label] = label_counts.get(label, 0) + 1
            for label, count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[:15]:
                f.write(f"  {standardize_label(label)}: frequency={count}\n")
            f.write("\n")
    print(f"Saved summary: ce_zr_cluster_heatmap_summary.txt")
