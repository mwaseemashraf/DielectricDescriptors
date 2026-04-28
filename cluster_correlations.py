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


def min_image_vector(vec, inv_latt, lattice):
    frac = vec.dot(inv_latt)
    frac -= np.round(frac)
    return frac.dot(lattice.T)


def cluster_geometry_label(positions, inv_latt, lattice, order):
    distances = []
    for i, j in combinations(range(order), 2):
        d = min_image_dist(positions[j] - positions[i], inv_latt, lattice)
        distances.append(round(d, 5))
    return tuple(sorted(distances))


def cluster_corr_value(sigmas):
    return np.prod(sigmas)


def compute_cluster_correlation_vector(lattice, positions, sigmas, orders=(2, 3, 4)):
    inv_latt = np.linalg.inv(lattice.T)
    labels = {order: {} for order in orders}

    for order in orders:
        for cluster in combinations(range(len(positions)), order):
            cluster_positions = positions[list(cluster)]
            label = cluster_geometry_label(cluster_positions, inv_latt, lattice, order)
            corr = cluster_corr_value([sigmas[i] for i in cluster])
            if label not in labels[order]:
                labels[order][label] = []
            labels[order][label].append(corr)

    correlation_vectors = {}
    for order in orders:
        correlation_vectors[order] = {
            "labels": sorted(labels[order].keys()),
            "values": [np.mean(labels[order][label]) for label in sorted(labels[order].keys())],
            "counts": [len(labels[order][label]) for label in sorted(labels[order].keys())],
        }
    return correlation_vectors


def standardize_label(label):
    return ",".join(f"{x:.3f}" for x in label)


def plot_correlations(structure_names, all_correlation_vectors, order, output_path):
    # Build the union of all geometry labels across structures
    all_labels = set()
    for corr in all_correlation_vectors:
        all_labels.update(corr[order]["labels"])
    labels = sorted(all_labels)

    # Create aligned vectors for each structure
    aligned_values = []
    for corr in all_correlation_vectors:
        value_map = {label: value for label, value in zip(corr[order]["labels"], corr[order]["values"])}
        aligned_values.append([value_map.get(label, np.nan) for label in labels])

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    for values, name in zip(aligned_values, structure_names):
        plt.plot(x, values, marker="o", label=name)

    plt.xticks(x, [standardize_label(l) for l in labels], rotation=45, ha="right", fontsize=8)
    plt.xlabel(f"Cluster geometry label (order={order})")
    plt.ylabel("Average cluster correlation")
    plt.title(f"Order-{order} cluster correlations for Ce/Zr doping sites")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    poscar_files = sorted(
        [f for f in os.listdir(ROOT) if f.lower().startswith("poscar") or f.lower().endswith("vasp")],
        key=lambda x: int("".join(filter(str.isdigit, x))) if any(ch.isdigit() for ch in x) else x,
    )

    structure_names = []
    all_correlation_vectors = []

    for filename in poscar_files:
        path = os.path.join(ROOT, filename)
        lattice, species_list, cart_coords = parse_poscar(path)

        doping_indices = [i for i, sp in enumerate(species_list) if sp.lower() in {"ce", "zr"}]
        if not doping_indices:
            continue

        doping_positions = cart_coords[doping_indices]
        sigmas = np.array([1 if species_list[i].lower() == "ce" else -1 for i in doping_indices])

        corr_vectors = compute_cluster_correlation_vector(lattice, doping_positions, sigmas, orders=(2, 3, 4))
        structure_names.append(filename)
        all_correlation_vectors.append(corr_vectors)

    if not all_correlation_vectors:
        raise RuntimeError("No doping structures found in training_set")

    for order in (2, 3, 4):
        plot_path = os.path.join(os.path.dirname(__file__), f"ce_zr_cluster_order_{order}.png")
        plot_correlations(structure_names, all_correlation_vectors, order, plot_path)
        print(f"Saved order-{order} cluster correlation plot to: {plot_path}")

    print("Cluster correlation summary:")
    for i, filename in enumerate(structure_names):
        print(f"\nStructure: {filename}")
        for order in (2, 3, 4):
            labels = all_correlation_vectors[i][order]["labels"]
            values = all_correlation_vectors[i][order]["values"]
            counts = all_correlation_vectors[i][order]["counts"]
            print(f"  Order {order}: {len(labels)} unique cluster types")
            for label, value, count in zip(labels, values, counts):
                print(f"    {standardize_label(label)} -> avg corr={value:.3f}, count={count}")
