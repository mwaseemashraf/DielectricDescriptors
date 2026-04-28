import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "labeled_poscars")
if not os.path.isdir(ROOT):
    raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

poscar_files = sorted(
    [f for f in os.listdir(ROOT) if f.lower().startswith("poscar") or f.lower().endswith("vasp")],
    key=lambda x: int("".join(filter(str.isdigit, x))) if any(ch.isdigit() for ch in x) else x,
)
if not poscar_files:
    raise RuntimeError("No POSCAR files found in labeled_poscars")


def parse_poscar(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [line.strip() for line in fh if line.strip()]
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


def min_image_vector(dvec, inv_latt, lattice):
    frac = dvec.dot(inv_latt)
    frac -= np.round(frac)
    return frac.dot(lattice.T)


def neighbor_environment(ce_pos, other_positions, inv_latt, lattice, max_neighbors=12):
    diffs = other_positions - ce_pos
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
        mask = distances <= cutoff
    else:
        mask = np.arange(min(max_neighbors, len(distances)))

    if mask.dtype == bool:
        selected = np.where(mask)[0]
    else:
        selected = mask
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
    eig_mean = np.mean(eigvals)
    eig_std = np.std(eigvals)
    shape_anisotropy = eig_std / np.maximum(eig_mean, 1e-8)

    angles = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            cosang = np.dot(normed[i], normed[j])
            cosang = np.clip(cosang, -1.0, 1.0)
            angles.append(np.arccos(cosang))
    angles = np.array(angles)
    angle_mean = np.degrees(np.mean(angles)) if len(angles) else np.nan
    angle_std = np.degrees(np.std(angles)) if len(angles) else np.nan

    score = 1.0 / (1.0 + dist_var + shape_anisotropy + angle_std / 180.0)
    return score, angle_mean, angle_std


def compute_angle_descriptor(angle_std_deg):
    if np.isnan(angle_std_deg):
        return np.nan
    return 1.0 / (1.0 + angle_std_deg / 20.0)


results = []
for filename in poscar_files:
    path = os.path.join(ROOT, filename)
    lattice, species_list, cart_coords = parse_poscar(path)
    inv_latt = np.linalg.inv(lattice.T)

    ce_indices = [i for i, sp in enumerate(species_list) if sp.lower().startswith("ce")]
    o_indices = [i for i, sp in enumerate(species_list) if sp.lower().startswith("o")]

    if not ce_indices:
        continue
    if not o_indices:
        raise RuntimeError(f"No oxygen atoms found in {filename}")

    ce_pos = cart_coords[ce_indices[0]]
    o_positions = cart_coords[o_indices]
    o_distances, o_vectors = neighbor_environment(ce_pos, o_positions, inv_latt, lattice, max_neighbors=12)
    symmetry_score, angle_mean_deg, angle_std_deg = compute_site_symmetry(o_vectors)
    angle_descriptor = compute_angle_descriptor(angle_std_deg)

    results.append(
        {
            "file": filename,
            "ce_count": len(ce_indices),
            "o_neighbors": len(o_vectors),
            "o_mean_dist": float(np.mean(o_distances)) if len(o_distances) else np.nan,
            "o_dist_std": float(np.std(o_distances)) if len(o_distances) else np.nan,
            "symmetry_score": float(symmetry_score),
            "angle_mean_deg": float(angle_mean_deg),
            "angle_std_deg": float(angle_std_deg),
            "angle_descriptor": float(angle_descriptor),
        }
    )

if not results:
    raise RuntimeError("No Ce environments were analyzed")

results.sort(key=lambda x: int("".join(filter(str.isdigit, x["file"]))) if any(ch.isdigit() for ch in x["file"]) else x["file"])

files = [r["file"] for r in results]
indices = list(range(len(results)))
coord_nums = [r["o_neighbors"] for r in results]
scores = [r["symmetry_score"] for r in results]
angle_stds = [r["angle_std_deg"] for r in results]
angle_desc = [r["angle_descriptor"] for r in results]
mean_dists = [r["o_mean_dist"] for r in results]
std_dists = [r["o_dist_std"] for r in results]

plt.figure(figsize=(12, 8))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.bar(indices, coord_nums, alpha=0.4, color="tab:blue", label="O coordination number")
ax2.plot(indices, scores, "o--", color="tab:red", label="Symmetry score")
ax1.set_xlabel("Structure file")
ax1.set_ylabel("O coordination number", color="tab:blue")
ax2.set_ylabel("Symmetry score", color="tab:red")
ax1.set_xticks(indices)
ax1.set_xticklabels(files, rotation=45, ha="right", fontsize=8)
ax1.set_title("Ce coordination number vs. local symmetry score")
ax1.grid(alpha=0.2)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.tight_layout()
coord_plot_path = os.path.join(os.path.dirname(__file__), "ce_coord_vs_symmetry.png")
plt.savefig(coord_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
plt.scatter(angle_stds, scores, c=coord_nums, cmap="viridis", s=100, edgecolors="k")
for i, file in enumerate(files):
    plt.text(angle_stds[i], scores[i] + 0.01, file, fontsize=8, ha="center", va="bottom")
plt.xlabel("Ce-O angle std dev (deg)")
plt.ylabel("Symmetry score")
plt.title("Ce local symmetry vs. Ce-O bond-angle dispersion")
plt.colorbar(label="O coordination number")
plt.grid(alpha=0.2)
plt.tight_layout()
angle_plot_path = os.path.join(os.path.dirname(__file__), "ce_angle_descriptor.png")
plt.savefig(angle_plot_path, dpi=300)
plt.close()

print("Computed Ce site-symmetry descriptors:")
for r in results:
    print(
        f"{r['file']}: o_neighbors={r['o_neighbors']}, mean_O_dist={r['o_mean_dist']:.4f}, "
        f"std_O_dist={r['o_dist_std']:.4f}, symmetry_score={r['symmetry_score']:.4f}, "
        f"angle_std={r['angle_std_deg']:.2f} deg, angle_descriptor={r['angle_descriptor']:.4f}"
    )
print("\nPlots saved to:")
print(" -", coord_plot_path)
print(" -", angle_plot_path)
print(" -", os.path.join(os.path.dirname(__file__), "ce_site_symmetry.png"))
