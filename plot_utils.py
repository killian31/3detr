import os
import re

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")


def extract_metrics_from_file(filepath):
    """Extract nqueries, mAP0.25, and mAP0.50 from a given results file."""
    with open(filepath, "r") as file:
        content = file.readlines()

    match = re.search(r"mAP0.25, mAP0.50:\s*([\d\.]+),\s*([\d\.]+)", content[0])
    if match:
        mAP0_25, mAP0_50 = float(match.group(1)), float(match.group(2))
        return mAP0_25, mAP0_50
    return None, None


def extract_ar_from_file(filepath):
    """Extract AR0.25 and AR0.50 from a given results file."""
    with open(filepath, "r") as file:
        content = file.readlines()

    match = re.search(r"AR0.25, AR0.50:\s*([\d\.]+),\s*([\d\.]+)", content[1])
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def plot_map_evolution(directory, save_filename):
    """Find result files, extract nqueries and mAP values, and plot the evolution."""
    files = [f for f in os.listdir(directory) if re.match(r"test_results_\d+\.txt", f)]

    results = []
    for file in files:
        match = re.search(r"test_results_(\d+)\.txt", file)
        if match:
            nqueries = int(match.group(1))
            mAP0_25, mAP0_50 = extract_metrics_from_file(os.path.join(directory, file))
            if mAP0_25 is not None and mAP0_50 is not None:
                results.append((nqueries, mAP0_25, mAP0_50))

    results.sort(key=lambda x: x[0])

    nqueries_vals = np.array([x[0] for x in results])
    mAP0_25_vals = np.array([x[1] for x in results])
    mAP0_50_vals = np.array([x[2] for x in results])

    plt.plot(
        nqueries_vals,
        mAP0_25_vals,
        color="#1f77b4",
        label="mAP@0.25",
        marker="o",
        linestyle="-",
    )
    plt.plot(
        nqueries_vals,
        mAP0_50_vals,
        color="#ff7f0e",
        label="mAP@0.50",
        marker="s",
        linestyle="--",
    )

    plt.xticks(nqueries_vals)

    plt.xlabel("Number of test queries", fontsize=14)
    plt.ylabel("Mean Average Precision (mAP)", fontsize=14)
    plt.title("AP on SUN RGB-D", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ar_evolution(directory, save_filename):
    """Plot AR@0.25 and AR@0.50 evolution across different nqueries."""
    files = [f for f in os.listdir(directory) if re.match(r"test_results_\d+\.txt", f)]

    results = []
    for file in files:
        match = re.search(r"test_results_(\d+)\.txt", file)
        if match:
            nqueries = int(match.group(1))
            AR0_25, AR0_50 = extract_ar_from_file(os.path.join(directory, file))
            if AR0_25 is not None and AR0_50 is not None:
                results.append((nqueries, AR0_25, AR0_50))

    # Sort results by nqueries
    results.sort(key=lambda x: x[0])

    # Extract x (nqueries) and y values (AR0.25, AR0.50)
    nqueries_vals = np.array([x[0] for x in results])
    AR0_25_vals = np.array([x[1] for x in results])
    AR0_50_vals = np.array([x[2] for x in results])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    plt.plot(
        nqueries_vals,
        AR0_25_vals,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        label="AR@0.25",
    )
    plt.plot(
        nqueries_vals,
        AR0_50_vals,
        marker="s",
        linestyle="--",
        color="#ff7f0e",
        label="AR@0.50",
    )

    plt.xlabel("Number of Queries", fontsize=14)
    plt.ylabel("Average Recall (AR)", fontsize=14)
    plt.title("AR on SUN RGB-D", fontsize=16, fontweight="bold")

    plt.xticks(nqueries_vals)  # Ensure x-axis is aligned with `nqueries`
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3, color="gray")

    # Save figure
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()


def extract_classwise_ap(filepath, iou_thresh):
    """Extract per-class AP values for a given IOU threshold (0.25 or 0.50) from a results file."""
    with open(filepath, "r") as file:
        content = file.readlines()

    iou_section = f"IOU Thresh={iou_thresh}"
    start_index = next(
        (i for i, line in enumerate(content) if iou_section in line), None
    )
    if start_index is None:
        return None  # IOU threshold section not found

    classwise_ap = {}
    for line in content[start_index + 1 :]:
        if "Average Precision" in line:
            match = re.match(r"(\w+) Average Precision:\s*([\d\.]+)", line)
            if match:
                class_name, ap_value = match.groups()
                classwise_ap[class_name] = float(ap_value)
        elif "Recall" in line:
            break

    return classwise_ap


def extract_classwise_ar(filepath, iou_thresh):
    """Extract per-class Recall values for a given IOU threshold (0.25 or 0.50)."""
    with open(filepath, "r") as file:
        content = file.readlines()

    iou_section = f"IOU Thresh={iou_thresh}"
    start_index = next(
        (i for i, line in enumerate(content) if iou_section in line), None
    )
    if start_index is None:
        return None  # IOU threshold section not found

    classwise_ar = {}
    for line in content[start_index + 1 :]:
        if "Recall" in line:
            match = re.match(r"(\w+) Recall:\s*([\d\.]+)", line)
            if match:
                class_name, ar_value = match.groups()
                classwise_ar[class_name] = float(ar_value)
        elif "IOU Thresh" in line:
            break  # Stop at the next IOU threshold

    return classwise_ar


def plot_classwise_ap(directory, save_filename, iou_thresh=0.25):
    """Plot the evolution of per-class AP across nqueries, for a given IOU threshold (0.25 or 0.50)."""
    files = [f for f in os.listdir(directory) if re.match(r"test_results_\d+\.txt", f)]

    results = {}
    for file in files:
        match = re.search(r"test_results_(\d+)\.txt", file)
        if match:
            nqueries = int(match.group(1))
            classwise_ap = extract_classwise_ap(
                os.path.join(directory, file), iou_thresh
            )
            if classwise_ap:
                results[nqueries] = classwise_ap

    sorted_nqueries = sorted(results.keys())

    all_classes = list(next(iter(results.values())).keys())

    ap_data = {
        cls: [results[nq].get(cls, 0) for nq in sorted_nqueries] for cls in all_classes
    }

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_classes)))

    for cls, color in zip(all_classes, colors):
        plt.plot(
            sorted_nqueries,
            ap_data[cls],
            marker="o",
            linestyle="-",
            label=cls,
            color=color,
        )

    plt.xlabel("Number of Queries", fontsize=14)
    plt.ylabel(f"Average Precision (AP@{iou_thresh})", fontsize=14)
    plt.title(
        f"Per-Class AP Evolution on SUN RGB-D (IOU={iou_thresh})",
        fontsize=16,
        fontweight="bold",
    )

    plt.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.xticks(sorted_nqueries)
    plt.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_classwise_ar(directory, save_filename, iou_thresh=0.25):
    """Plot the evolution of per-class Recall across nqueries, for a given IOU threshold (0.25 or 0.50)."""
    files = [f for f in os.listdir(directory) if re.match(r"test_results_\d+\.txt", f)]

    results = {}
    for file in files:
        match = re.search(r"test_results_(\d+)\.txt", file)
        if match:
            nqueries = int(match.group(1))
            classwise_ar = extract_classwise_ar(
                os.path.join(directory, file), iou_thresh
            )
            if classwise_ar:
                results[nqueries] = classwise_ar

    sorted_nqueries = sorted(results.keys())

    all_classes = list(next(iter(results.values())).keys())

    ar_data = {
        cls: [results[nq].get(cls, 0) for nq in sorted_nqueries] for cls in all_classes
    }

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_classes)))

    for cls, color in zip(all_classes, colors):
        plt.plot(
            sorted_nqueries,
            ar_data[cls],
            marker="o",
            linestyle="-",
            label=cls,
            color=color,
        )

    plt.xlabel("Number of Queries", fontsize=14)
    plt.ylabel(f"Average Recall (AR@{iou_thresh})", fontsize=14)
    plt.title(
        f"Per-Class AR Evolution on SUN RGB-D (IOU={iou_thresh})",
        fontsize=16,
        fontweight="bold",
    )

    plt.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.xticks(sorted_nqueries)
    plt.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_map_evolution("./results_test_nqueries/", "map_evolution_nqueries.png")
    plot_classwise_ap(
        "./results_test_nqueries/", "classwise_ap0.50_evolution.png", iou_thresh=0.50
    )
    plot_ar_evolution("./results_test_nqueries/", "ar_evolution_nqueries.png")
    plot_classwise_ar(
        "./results_test_nqueries/", "classwise_ar0.50_evolution.png", iou_thresh=0.50
    )
    plot_classwise_ar(
        "./results_test_nqueries/", "classwise_ar0.25_evolution.png", iou_thresh=0.25
    )
