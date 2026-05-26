"""
T-block keypoint detection from lerobot/pusht images via contour detection.

Strategy: detect the T-block as the unique *achromatic* (low chroma),
medium-brightness, non-blue region.  We do NOT try to exclude the goal zone:
the T-block stays gray even when it sits on the green goal, and any
goal-exclusion criterion eats into the mask at the overlap boundary after the
512→96 downscale blurs colors.

Pixel analysis of actual dataset images:
  T-block       : chroma 17–60, brightness 130–195, B-R < 50
  Goal zone     : chroma 70–100, brightness 170–185   → excluded by chroma
  Agent (blue)  : chroma 150+, B-R ≈ 158              → excluded by B-R
  Background    : brightness > 230                    → excluded by brightness

Run:  python detect_keypoints.py
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID    = "lerobot/pusht"
IMG_SIZE   = 96
WORLD_SIZE = 512
SCALE      = WORLD_SIZE / IMG_SIZE   # 96-px → 512-world


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _t_block_mask(image_rgb: np.ndarray) -> np.ndarray:
    r = image_rgb[:, :, 0].astype(np.int16)
    g = image_rgb[:, :, 1].astype(np.int16)
    b = image_rgb[:, :, 2].astype(np.int16)

    mask_blue  = (b - r) > 100          # agent: B-R≈157, T-block: B-R≈34
    mask_green = (g - r) > 40           # goal:  G-R≈94,  T-block: G-R≈17
    mask_white = (r > 240) & (g > 240) & (b > 240)

    mask = (mask_blue == False) & (mask_green == False) & (mask_white == False)

    # morhological opening to clean up noise
    kernel_1 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_1)
    return mask * 255


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------

def detect_keypoints(image_rgb: np.ndarray, n_pts: int = 16):
    """
    Detect T-block shape from a 96×96 RGB image by resampling the contour
    to a fixed number of equidistant points.

    Returns
    -------
    keypoints : (n_pts, 2) float32 in [0, 512] world coords, consistently
                ordered starting from the rightmost boundary point.
    mask      : (96, 96) uint8 binary mask.
    """
    mask = _t_block_mask(image_rgb)

    # CHAIN_APPROX_NONE gives every boundary pixel for smooth resampling
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None, mask

    pts = c.reshape(-1, 2).astype(np.float32)

    # Arc-length parameterisation → resample to n_pts equidistant points
    diffs   = np.diff(pts, axis=0, prepend=pts[-1:])
    arc     = np.cumsum(np.linalg.norm(diffs, axis=1))
    arc     = arc / arc[-1]                          # normalise to [0, 1]
    t       = np.linspace(0, 1, n_pts, endpoint=False)
    xs      = np.interp(t, arc, pts[:, 0])
    ys      = np.interp(t, arc, pts[:, 1])
    resampled = np.stack([xs, ys], axis=1)

    return resampled * SCALE, mask  # return in world coords [0, 512]


def block_pose(image_rgb: np.ndarray):
    """
    Compact (cx, cy, angle_rad) block pose in world coords.
    Uses the minimum-area bounding rectangle of the T-block mask.
    Useful as a 3D alternative to the full keypoint set.
    """
    mask = _t_block_mask(image_rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    (cx, cy), _, angle_deg = cv2.minAreaRect(c)
    return np.array([cx * SCALE, cy * SCALE, np.deg2rad(angle_deg)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(ds, num_samples: int = 6, save_path: str = "keypoint_detection.png"):
    """
    4-column grid per row:
      original | t-block mask | masked pixels on image | keypoints overlay
    """
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)
    ncols = 4
    fig, axes = plt.subplots(num_samples, ncols, figsize=(ncols * 3, 3.5 * num_samples))

    col_titles = ["original", "T-block mask", "masked pixels", "keypoints"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight="bold")

    for row, idx in enumerate(indices):
        sample   = ds[int(idx)]
        img_np   = (sample["observation.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        agent_w  = sample["observation.state"].numpy()
        agent_px = agent_w / SCALE

        mask    = _t_block_mask(img_np)
        kp_w, _ = detect_keypoints(img_np)

        # col 0: original
        axes[row, 0].imshow(img_np)
        axes[row, 0].axis("off")

        # col 1: binary mask
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].axis("off")

        # col 2: masked pixels shown on image
        overlay = img_np.copy()
        overlay[mask == 0] = 255
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

        # col 3: keypoints
        axes[row, 3].imshow(img_np)
        if kp_w is not None:
            kp_px = kp_w / SCALE
            axes[row, 3].scatter(kp_px[:, 0], kp_px[:, 1], c="red", s=40, zorder=5)
            for i, (x, y) in enumerate(kp_px):
                axes[row, 3].text(x, y, str(i), color="white",
                                  fontsize=7, ha="center", va="center", zorder=6)
            axes[row, 3].add_patch(
                plt.Polygon(kp_px, fill=False, edgecolor="red", linewidth=1.5))
            axes[row, 3].set_title(f"{len(kp_w)} pts", fontsize=8)
        else:
            axes[row, 3].set_title("FAILED", fontsize=8, color="red")

        axes[row, 3].scatter([agent_px[0]], [agent_px[1]],
                             c="cyan", s=60, marker="+", linewidths=2, zorder=7)
        axes[row, 3].axis("off")

    handles = [
        mpatches.Patch(color="red",  label="T-block keypoints"),
        mpatches.Patch(color="cyan", label="agent pos (from state)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def detection_stats(ds, num_samples: int = 300):
    from tqdm import tqdm

    n      = min(num_samples, len(ds))
    idxs   = np.random.choice(len(ds), n, replace=False)
    ok     = 0
    counts = []

    for idx in tqdm(idxs, desc="Evaluating"):
        sample = ds[int(idx)]
        img_np = (sample["observation.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kp, _  = detect_keypoints(img_np)
        if kp is not None:
            ok += 1
            counts.append(len(kp))

    print(f"\nDetection rate : {ok}/{n}  ({100*ok/n:.1f}%)")
    if counts:
        vals, freq = np.unique(counts, return_counts=True)
        print("Corner count distribution:")
        for v, f in zip(vals, freq):
            bar = "█" * int(30 * f / len(counts))
            print(f"  {v:2d} corners : {bar} {f} ({100*f/len(counts):.1f}%)")
        print(f"\n  Mean ± std : {np.mean(counts):.2f} ± {np.std(counts):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset (first 5 episodes) ...")
    ds = LeRobotDataset(REPO_ID, episodes=list(range(5)))

    print("Visualising detections (8 frames) ...")
    visualize(ds, num_samples=8)

    print("Detection stats over 300 frames ...")
    detection_stats(ds, num_samples=300)


if __name__ == "__main__":
    main()
