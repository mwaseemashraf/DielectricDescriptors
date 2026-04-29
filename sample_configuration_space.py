import csv
import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from analyze_vasp_dielectric_descriptors import find_structure_runs, parse_poscar, min_image_vector


ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "vasp_descriptor_results")
OUTPUT_DIR = os.path.join(ROOT, "configuration_sampling_results")


def numeric_key(name):
    match = re.search(r"POSCAR\.(\d+)$", name)
    return int(match.group(1)) if match else name


def read_descriptor_table():
    path = os.path.join(RESULTS_DIR, "descriptor_table.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        clean = {"structure": row["structure"]}
        for key, value in row.items():
            if key == "structure":
                continue
            try:
                clean[key] = float(value) if value != "" else np.nan
            except ValueError:
                clean[key] = np.nan
        out[row["structure"]] = clean
    return out


def cation_pair_shell_vector(poscar_path, bin_width=0.25, min_pair_distance=1.0):
    lattice, species, cart = parse_poscar(poscar_path)
    inv_latt = np.linalg.inv(lattice.T)
    cation_indices = [i for i, sp in enumerate(species) if sp.lower() in {"ce", "zr"}]
    sigmas = {i: 1.0 if species[i].lower() == "ce" else -1.0 for i in cation_indices}

    shell_values = {}
    shell_counts = {}
    for i, j in combinations(cation_indices, 2):
        dist = np.linalg.norm(min_image_vector(cart[j] - cart[i], inv_latt, lattice))
        if dist < min_pair_distance:
            continue
        shell = round(round(dist / bin_width) * bin_width, 3)
        shell_values.setdefault(shell, []).append(sigmas[i] * sigmas[j])
        shell_counts[shell] = shell_counts.get(shell, 0) + 1

    return {shell: float(np.mean(values)) for shell, values in shell_values.items()}, shell_counts


def build_pair_matrix(runs):
    vectors = []
    count_vectors = []
    structures = []
    for run in runs:
        initial_poscar = os.path.join(run["path"], run["structure"])
        if not os.path.exists(initial_poscar):
            initial_poscar = os.path.join(ROOT, "training_set", run["structure"])
        vector, counts = cation_pair_shell_vector(initial_poscar)
        structures.append(run["structure"])
        vectors.append(vector)
        count_vectors.append(counts)

    shells = sorted({shell for vector in vectors for shell in vector})
    matrix = np.array([[vector.get(shell, 0.0) for shell in shells] for vector in vectors], dtype=float)
    count_matrix = np.array([[counts.get(shell, 0.0) for shell in shells] for counts in count_vectors], dtype=float)
    return structures, shells, matrix, count_matrix


def pca(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std == 0] = 1.0
    z = (matrix - mean) / std
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = u * s
    explained = (s**2) / np.sum(s**2)
    loadings = vt.T
    return scores, explained, loadings


def kmeans(points, k=4, max_iter=100):
    # Deterministic farthest-point initialization.
    centers = [points[np.argmin(points[:, 0])]]
    while len(centers) < k:
        distances = np.min([np.sum((points - center) ** 2, axis=1) for center in centers], axis=0)
        centers.append(points[np.argmax(distances)])
    centers = np.array(centers, dtype=float)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        new_labels = np.argmin(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = np.mean(points[labels == i], axis=0)

    representatives = []
    for i in range(k):
        members = np.where(labels == i)[0]
        if len(members) == 0:
            continue
        dist = np.linalg.norm(points[members] - centers[i], axis=1)
        representatives.append(int(members[np.argmin(dist)]))
    return labels, centers, representatives


def plot_pca_sampling(structures, scores, explained, labels, representatives, response, output_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    colors = plt.get_cmap("tab10")(labels)
    ax.scatter(scores[:, 0], scores[:, 1], s=72, c=colors, edgecolors="black", linewidths=0.7)
    ax.scatter(
        scores[representatives, 0],
        scores[representatives, 1],
        s=210,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="cluster representative",
    )
    for i, name in enumerate(structures):
        ax.annotate(name.replace("POSCAR.", ""), (scores[i, 0], scores[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Ce/Zr Pair-Correlation Configuration Space")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_response_maps(structures, scores, explained, response_rows, output_path):
    panels = [
        ("eps_trace_avg", "Ionic dielectric average"),
        ("optical_softness_inv_freq2", "Soft-mode descriptor"),
        ("lattice_length_cv", "Lattice anisotropy"),
        ("mean_neighbor_distance_variation", "Neighbor-distance variation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), squeeze=False)
    for ax, (key, title) in zip(axes.ravel(), panels):
        values = np.array([response_rows[name].get(key, np.nan) for name in structures], dtype=float)
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=values, cmap="viridis", s=72, edgecolors="black", linewidths=0.6)
        for i, name in enumerate(structures):
            ax.annotate(name.replace("POSCAR.", ""), (scores[i, 0], scores[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="0.8", linewidth=0.7)
        ax.axvline(0, color="0.8", linewidth=0.7)
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pair_correlation_heatmap(structures, shells, matrix, labels, output_path):
    order = np.lexsort((np.arange(len(structures)), labels))
    ordered_matrix = matrix[order]
    ordered_names = [structures[i] for i in order]
    ordered_labels = labels[order]

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    im = ax.imshow(ordered_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels([name.replace("POSCAR.", "") for name in ordered_names])
    ax.set_xticks(range(len(shells)))
    ax.set_xticklabels([f"{shell:.2f}" for shell in shells], rotation=45, ha="right")
    ax.set_xlabel("Ce/Zr cation-pair shell distance (A)")
    ax.set_ylabel("Starting structure")
    ax.set_title("Shell-Binned Ce/Zr Pair-Correlation Vectors")
    for row_index, cluster in enumerate(ordered_labels):
        ax.text(-0.8, row_index, f"C{cluster + 1}", va="center", ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Average pair correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_loading_bars(shells, loadings, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    x = np.arange(len(shells))
    for pc_index, ax in enumerate(axes):
        ax.bar(x, loadings[:, pc_index], color="0.25")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{shell:.2f}" for shell in shells], rotation=45, ha="right")
        ax.set_xlabel("Pair shell distance (A)")
        ax.set_title(f"PC{pc_index + 1} loadings")
    axes[0].set_ylabel("Loading")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_outputs(structures, shells, matrix, scores, explained, labels, representatives, response_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUTPUT_DIR, "configuration_sampling_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "structure",
            "cluster",
            "representative",
            "pc1",
            "pc2",
            "eps_trace_avg",
            "optical_softness_inv_freq2",
            "lattice_length_cv",
            "mean_neighbor_distance_variation",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        rep_set = set(representatives)
        for i, name in enumerate(structures):
            response = response_rows[name]
            writer.writerow(
                {
                    "structure": name,
                    "cluster": int(labels[i] + 1),
                    "representative": "yes" if i in rep_set else "",
                    "pc1": scores[i, 0],
                    "pc2": scores[i, 1],
                    "eps_trace_avg": response.get("eps_trace_avg", np.nan),
                    "optical_softness_inv_freq2": response.get("optical_softness_inv_freq2", np.nan),
                    "lattice_length_cv": response.get("lattice_length_cv", np.nan),
                    "mean_neighbor_distance_variation": response.get("mean_neighbor_distance_variation", np.nan),
                }
            )

    vectors_path = os.path.join(OUTPUT_DIR, "pair_correlation_shell_vectors.csv")
    with open(vectors_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["structure"] + [f"shell_{shell:.2f}A" for shell in shells]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in zip(structures, matrix):
            out = {"structure": name}
            out.update({f"shell_{shell:.2f}A": value for shell, value in zip(shells, row)})
            writer.writerow(out)

    notes_path = os.path.join(OUTPUT_DIR, "configuration_sampling_notes.md")
    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Configuration-space sampling analysis\n\n"
            "- Primary descriptor: shell-binned Ce/Zr cation pair-correlation vectors from the starting POSCAR structures.\n"
            "- Dimensionality reduction: PCA of standardized pair-correlation vectors.\n"
            "- Representative structures: deterministic k-means clusters in PC1-PC2 space; circled points are nearest to cluster centroids.\n"
            "- Response overlays use relaxed/DFPT quantities only after sampling: ionic dielectric average, soft-mode descriptor, lattice anisotropy, and neighbor-distance variation.\n"
            "- This supports a sampling claim, not a standalone dielectric-prediction claim.\n"
        )


def main():
    runs = find_structure_runs()
    response_rows = read_descriptor_table()
    structures, shells, matrix, _ = build_pair_matrix(runs)
    scores, explained, loadings = pca(matrix)
    labels, _, representatives = kmeans(scores[:, :2], k=4)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_pca_sampling(
        structures,
        scores,
        explained,
        labels,
        representatives,
        response_rows,
        os.path.join(OUTPUT_DIR, "configuration_space_pca_clusters.png"),
    )
    plot_response_maps(
        structures,
        scores,
        explained,
        response_rows,
        os.path.join(OUTPUT_DIR, "configuration_space_response_overlays.png"),
    )
    plot_pair_correlation_heatmap(
        structures,
        shells,
        matrix,
        labels,
        os.path.join(OUTPUT_DIR, "pair_correlation_shell_heatmap.png"),
    )
    plot_loading_bars(
        shells,
        loadings,
        os.path.join(OUTPUT_DIR, "pair_correlation_pca_loadings.png"),
    )
    write_outputs(structures, shells, matrix, scores, explained, labels, representatives, response_rows)

    rep_names = ", ".join(structures[i] for i in representatives)
    print(f"Wrote configuration sampling outputs to {OUTPUT_DIR}")
    print(f"Representative structures: {rep_names}")


if __name__ == "__main__":
    main()
