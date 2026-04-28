import os
import numpy as np
import matplotlib.pyplot as plt

root = os.path.join(os.path.dirname(__file__), "training_set")
if not os.path.isdir(root):
    raise FileNotFoundError(f"Dataset folder not found: {root}")

poscar_files = sorted(
    [f for f in os.listdir(root) if f.lower().startswith("poscar") or f.lower().endswith("vasp")],
    key=lambda x: int("".join(filter(str.isdigit, x))) if any(ch.isdigit() for ch in x) else x,
)
if not poscar_files:
    raise RuntimeError("No POSCAR files found in training_set")

results = []

for filename in poscar_files:
    path = os.path.join(root, filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 9:
        raise ValueError(f"POSCAR file too short: {filename}")

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

    idx = 0
    ce_indices = []
    zr_indices = []
    o_indices = []
    for sp, count in zip(species, counts):
        if sp.lower().startswith("ce"):
            ce_indices.extend(range(idx, idx + count))
        elif sp.lower().startswith("zr"):
            zr_indices.extend(range(idx, idx + count))
        elif sp.lower().startswith("o"):
            o_indices.extend(range(idx, idx + count))
        idx += count

    n_ce = len(ce_indices)
    if n_ce == 0:
        continue

    ce_coords = cart_coords[ce_indices]
    inv_latt = np.linalg.inv(lattice.T)

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

    if n_ce >= 2:
        ce_dists = min_image_distance_matrix(ce_coords, ce_coords, lattice, inv_latt)
        ce_pairs = ce_dists[np.triu_indices(n_ce, k=1)]
        ce_ce_min = float(np.min(ce_pairs))
    else:
        # Only one Ce per cell: use the smallest nonzero lattice-translation length as a periodic Ce-Ce distance.
        translations = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if (i, j, k) == (0, 0, 0):
                        continue
                    vec = i * lattice[0] + j * lattice[1] + k * lattice[2]
                    translations.append(np.linalg.norm(vec))
        ce_ce_min = float(np.min(translations))

    zr_min = float(np.min(min_image_distance_matrix(ce_coords, cart_coords[zr_indices], lattice, inv_latt))) if zr_indices else np.nan
    o_min = float(np.min(min_image_distance_matrix(ce_coords, cart_coords[o_indices], lattice, inv_latt))) if o_indices else np.nan

    results.append(
        {
            "file": filename,
            "n_ce": n_ce,
            "ce_ce_min": ce_ce_min,
            "ce_zr_min": zr_min,
            "ce_o_min": o_min,
            "ce_position": tuple(ce_coords[0]) if len(ce_coords) == 1 else None,
        }
    )

if not results:
    raise RuntimeError("No Ce atoms found in any POSCAR file")

results.sort(key=lambda x: int("".join(filter(str.isdigit, x["file"]))) if any(ch.isdigit() for ch in x["file"]) else x["file"])

files = [r["file"] for r in results]
indices = list(range(len(results)))
ce_ce_mins = [r["ce_ce_min"] for r in results]
ce_zr_mins = [r["ce_zr_min"] for r in results]
ce_o_mins = [r["ce_o_min"] for r in results]

plt.figure(figsize=(12, 8))
plt.plot(indices, ce_ce_mins, "o-", label="Ce-Ce periodic min distance")
plt.plot(indices, ce_zr_mins, "s-", label="nearest Ce-Zr distance")
plt.plot(indices, ce_o_mins, "^-", label="nearest Ce-O distance")
plt.xticks(indices, files, rotation=45, ha="right", fontsize=8)
plt.xlabel("Structure file")
plt.ylabel("Distance (Å)")
plt.title("Ce-based distance descriptors across training_set")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "ce_descriptor_plot.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Computed Ce-based distance descriptors for POSCAR files:")
for r in results:
    print(
        f"{r['file']}: n_ce={r['n_ce']}, ce_ce_min={r['ce_ce_min']:.4f}, ce_zr_min={r['ce_zr_min']:.4f}, ce_o_min={r['ce_o_min']:.4f}"
    )
print("\nPlot saved to:", plot_path)
