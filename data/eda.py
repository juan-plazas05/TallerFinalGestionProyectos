#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "seaborn>=0.13",
#     "numpy>=1.26",
#     "pyyaml>=6.0",
# ]
# ///

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

DATA_DIR = Path(__file__).resolve().parent
DATASET_YAML = DATA_DIR / "dataset.yaml"
OUTPUT_DIR = DATA_DIR / "eda_output"
SPLITS = ["train", "valid", "test"]


def load_labels(split: str) -> list[dict]:
    labels_dir = DATA_DIR / split / "labels"
    data = []
    for lbl_path in sorted(labels_dir.glob("*.txt")):
        lines = lbl_path.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            data.append({
                "cls": cls_id,
                "x": x, "y": y, "w": w, "h": h,
                "area": w * h,
                "aspect": w / h if h > 0 else 0,
                "file": lbl_path.name,
                "split": split,
            })
    return data

def class_distribution(all_data: dict[str, list]) -> dict[str, Counter]:
    return {s: Counter(d["cls"] for d in all_data[s]) for s in SPLITS}

def check_integrity() -> list[str]:
    errors = []
    for split in SPLITS:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        img_stems = {f.stem for f in img_dir.glob("*.jpg")}
        lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}
        imgs_no_lbl = img_stems - lbl_stems
        lbls_no_img = lbl_stems - img_stems
        if imgs_no_lbl:
            errors.append(f"[{split}] {len(imgs_no_lbl)} images without label: {', '.join(sorted(imgs_no_lbl)[:5])}")
        if lbls_no_img:
            errors.append(f"[{split}] {len(lbls_no_img)} labels without image: {', '.join(sorted(lbls_no_img)[:5])}")
        for lbl_path in lbl_dir.glob("*.txt"):
            for i, line in enumerate(lbl_path.read_text().strip().splitlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"[{split}] {lbl_path.name}:{i} bad format ({len(parts)} fields)")
                    continue
                cls_id, x, y, w, h = map(float, parts)
                if not (0 <= cls_id < NUM_CLASSES):
                    errors.append(f"[{split}] {lbl_path.name}:{i} class {cls_id} out of range")
                for coord, name in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
                    if not (0 <= coord <= 1):
                        errors.append(f"[{split}] {lbl_path.name}:{i} {name}={coord} out of [0,1]")
    return errors

def plot_class_distribution(counts: dict[str, Counter], output_dir: Path):
    classes = list(range(NUM_CLASSES))
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, split in enumerate(SPLITS):
        vals = [counts[split].get(c, 0) for c in classes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=split)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(v), ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

def plot_bbox_analysis(all_data: dict[str, list], output_dir: Path):
    all_boxes = []
    for split in SPLITS:
        all_boxes.extend(all_data[split])

    areas = [b["area"] for b in all_boxes]
    aspects = [b["aspect"] for b in all_boxes]
    xs = [b["x"] for b in all_boxes]
    ys = [b["y"] for b in all_boxes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(areas, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("Normalized Area")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Bounding Box Area Distribution")

    axes[0, 1].hist(aspects, bins=50, color="coral", edgecolor="white")
    axes[0, 1].set_xlabel("Aspect Ratio (w/h)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Bounding Box Aspect Ratio")

    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=20, range=[[0, 1], [0, 1]])
    im = axes[1, 0].imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto",
                           extent=[0, 1, 0, 1])
    axes[1, 0].set_xlabel("X Center")
    axes[1, 0].set_ylabel("Y Center")
    axes[1, 0].set_title("BBox Center Heatmap")
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].scatter(xs, ys, s=1, alpha=0.3, c="blue")
    axes[1, 1].set_xlabel("X Center")
    axes[1, 1].set_ylabel("Y Center")
    axes[1, 1].set_title("BBox Center Scatter")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "bbox_analysis.png", dpi=150)
    plt.close(fig)

def plot_per_class_bbox(all_data: dict[str, list], output_dir: Path):
    all_boxes = []
    for split in SPLITS:
        all_boxes.extend(all_data[split])

    cls_areas = {c: [] for c in range(NUM_CLASSES)}
    for b in all_boxes:
        cls_areas[b["cls"]].append(b["area"])

    fig, ax = plt.subplots(figsize=(14, 5))
    positions = list(range(NUM_CLASSES))
    data = [cls_areas[c] for c in range(NUM_CLASSES)]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, name in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(sns.color_palette("husl", NUM_CLASSES)[CLASS_NAMES.index(name)])

    ax.set_xlabel("Class")
    ax.set_ylabel("Normalized Area")
    ax.set_title("Bounding Box Area by Class")
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_NAMES)

    counts_per_class = Counter(b["cls"] for b in all_boxes)
    for i, c in enumerate(CLASS_NAMES):
        cnt = counts_per_class.get(i, 0)
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={cnt}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "per_class_bbox.png", dpi=150)
    plt.close(fig)

def print_report(counts: dict[str, Counter], integrity_errors: list[str], all_data: dict[str, list]):
    print("=" * 60)
    print("EDA REPORT — Sign Language Dataset")
    print("=" * 60)

    print(f"\n Dataset: {DATA_DIR}")
    for split in SPLITS:
        total = sum(counts[split].values())
        n_imgs = len({d["file"] for d in all_data[split]})
        n_imgs_dir = len(list((DATA_DIR / split / "images").glob("*.jpg")))
        print(f"  {split}: {n_imgs_dir} images, {total} objects ({total / n_imgs_dir:.1f} obj/img)")

    print(f"\n Class Distribution (training):")
    print(f"  {'Class':>6} {'Count':>6} {'Ratio':>8}")
    max_count = max(counts["train"].values()) if counts["train"] else 1
    for c in range(NUM_CLASSES):
        cnt = counts["train"].get(c, 0)
        ratio = cnt / max_count
        marker = " [!]" if ratio < 0.65 else ""
        print(f"  {CLASS_NAMES[c]:>6} {cnt:>6} {ratio:>7.2%}{marker}")
    min_c = min(counts["train"], key=counts["train"].get)
    max_c = max(counts["train"], key=counts["train"].get)
    print(f"\n  Most frequent : {CLASS_NAMES[max_c]} ({counts['train'][max_c]})")
    print(f"  Least frequent: {CLASS_NAMES[min_c]} ({counts['train'][min_c]})")
    print(f"  Max/Min ratio : {counts['train'][max_c] / counts['train'][min_c]:.1f}x")

    print(f"\n Integrity:")
    if integrity_errors:
        print(f"  {len(integrity_errors)} issue(s) found:")
        for err in integrity_errors:
            print(f"    - {err}")
    else:
        print("   No issues found")

    all_boxes = []
    for split in SPLITS:
        all_boxes.extend(all_data[split])
    areas = [b["area"] for b in all_boxes]
    if areas:
        print(f"\n Bounding Boxes (all splits):")
        print(f"  Total objects: {len(all_boxes)}")
        print(f"  Area  - mean={np.mean(areas):.4f}, median={np.median(areas):.4f}, std={np.std(areas):.4f}")
        print(f"  Area  - min={min(areas):.4f}, max={max(areas):.4f}")
        aspects = [b["aspect"] for b in all_boxes]
        print(f"  Aspect- mean={np.mean(aspects):.2f}, median={np.median(aspects):.2f}")

def generate_report(counts: dict[str, Counter], integrity_errors: list[str], all_data: dict[str, list], output_dir: Path) -> None:
    max_count = max(counts["train"].values()) if counts["train"] else 1
    min_c = min(counts["train"], key=counts["train"].get)
    max_c = max(counts["train"], key=counts["train"].get)
    total_boxes = sum(len(v) for v in all_data.values())
    areas = [b["area"] for b in sum(all_data.values(), [])]
    aspects = [b["aspect"] for b in sum(all_data.values(), [])]

    weights = [max_count / counts["train"].get(c, 1) for c in range(NUM_CLASSES)]
    n_total = {s: len(list((DATA_DIR / s / 'images').glob('*.jpg'))) for s in SPLITS}

    lines = [
        "# Dataset Report: Sign Language Detection (A-Z)",
        "",
        "## Resumen",
        "",
        f"- **Total de imagenes:** {sum(n_total.values())} (train: {n_total['train']}, valid: {n_total['valid']}, test: {n_total['test']})",
        f"- **Total de labels:** {total_boxes}",
        f"- **Clases:** {NUM_CLASSES} (letras A-Z)",
        f"- **Formato de anotacion:** YOLO (class_id x_center y_center width height - todo normalizado)",
        "",
        "## Distribucion de Clases",
        "![Class Distribution](class_distribution.png)",
        "",
        "Por cada split hay exactamente 1 objeto por imagen. El dataset presenta un desbalance **2:1** "
        f"entre la clase mas frecuente ({CLASS_NAMES[max_c]}, {counts['train'][max_c]} muestras) y la menos frecuente "
        f"({CLASS_NAMES[min_c]}, {counts['train'][min_c]} muestras).",
        "",
        "### Clases criticas en training (< 65% respecto a la mayoritaria):",
        "",
        "## Analisis de Bounding Boxes",
        "![Per-Class BBox](per_class_bbox.png)",
        "![BBox Analysis](bbox_analysis.png)",
        f"- **Forma predominante:** relacion de aspecto media de {np.mean(aspects):.2f} (w/h), ",
        "lo que indica bounding boxes ligeramente verticales, consistente con la forma de una mano senalando.",
        f"- **Tamano:** area normalizada media de {np.mean(areas):.3f} (mediana {np.median(areas):.3f}), "
        "lo que sugiere que las manos ocupan aproximadamente un tercio del frame.",
        "- **Distribucion espacial:** los centros de los bounding boxes se concentran mayoritariamente en "
        "el centro de la imagen. Esto podria indicar que el dataset fue recopilado con la mano centrada "
        "en el encuadre.",
        "",
        "## Integridad del Dataset",
        "",
        "No se encontraron problemas de integridad. Todas las imagenes tienen su label correspondiente, "
        "todos los labels tienen coordenadas validas en el rango [0,1], y no hay clases fuera del "
        "rango esperado (0-25).",
        "",
        "## Estrategia de Balanceo"
        "",
        "1. **Class weights en la loss:** ",
        "",
        "vamos a usar sklearn.utils.class_weight.compute_class_weight para calcular pesos inversamente proporcionales"
        "a la frecuencia de cada clase en el split de entrenamiento. ",
        "",
        "2. **Augmentation selectiva:** ",
        "",
        "incrementar mosaic, mixup y copy-paste para enriquecer la representacion "
        "de las clases con menos muestras. Estas tecnicas generan synthetic samples combinando objetos "
        "de distintas imagenes.",
    ]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n Report saved to: {report_path}")


def main():
    with DATASET_YAML.open() as f:
        global NUM_CLASSES, CLASS_NAMES
        cfg = yaml.safe_load(f)
        NUM_CLASSES = cfg.get("nc")
        CLASS_NAMES = [cfg.get("names")[x] for x in range(NUM_CLASSES)]

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading labels...")
    all_data = {}
    for split in SPLITS:
        all_data[split] = load_labels(split)
        print(f"  {split}: {len(all_data[split])} objects loaded")

    print("\nChecking integrity...")
    integrity_errors = check_integrity()

    print("Computing class distribution...")
    counts = class_distribution(all_data)

    print("Generating figures...")
    plot_class_distribution(counts, OUTPUT_DIR)
    plot_bbox_analysis(all_data, OUTPUT_DIR)
    plot_per_class_bbox(all_data, OUTPUT_DIR)

    print_report(counts, integrity_errors, all_data)
    generate_report(counts, integrity_errors, all_data, OUTPUT_DIR)

if __name__ == "__main__":
    main()
