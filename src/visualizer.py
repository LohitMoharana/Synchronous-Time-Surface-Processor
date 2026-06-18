# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_time_surface(matrix, title, filename="output/heatmap.png"):
    """
    Generates a publication-ready heatmap image of the 8x8 matrix.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(6, 5))

    # Use a high-contrast colormap ('magma' or 'inferno' look great in papers)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="magma",
                     cbar_kws={'label': 'Intensity (Temporal Recency)'},
                     vmin=0, vmax=255)

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Sensor Grid X", fontsize=12)
    plt.ylabel("Sensor Grid Y", fontsize=12)

    # Save it at 300 DPI (the standard resolution required by IEEE/ACM journals)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Visualizer] Saved heatmap to {filename}")