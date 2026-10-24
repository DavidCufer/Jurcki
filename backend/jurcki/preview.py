"""Quick-look PNGs of the static layers: ``python -m jurcki.preview``."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, grid

CMAPS = {"elev": "terrain", "slope": "magma", "northness": "RdBu_r", "eastness": "RdBu_r",
         "tpi500": "RdBu_r", "tpi2000": "RdBu_r", "twi": "viridis_r", "tri": "magma",
         "treecover": "Greens", "grassland": "YlGn", "cropland": "YlOrBr",
         "builtup": "Greys", "forest_edge_dist": "PuOr"}
SYMMETRIC = {"northness", "eastness", "tpi500", "tpi2000", "forest_edge_dist"}


def main() -> None:
    out = config.INTERIM / "preview"
    out.mkdir(exist_ok=True)
    land = grid.load_mask()
    for name, cmap in CMAPS.items():
        path = config.INTERIM / "static" / f"{name}.tif"
        if not path.exists():
            continue
        a = np.where(land, grid.read_tif(path), np.nan)
        lo, hi = np.nanpercentile(a, [1, 99])
        if name in SYMMETRIC:
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        fig, ax = plt.subplots(figsize=(9, 6.5))
        im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.savefig(out / f"{name}.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(out / f"{name}.png")


if __name__ == "__main__":
    main()
