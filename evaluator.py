import cv2
import time
import re
import os
import json
import shutil
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, label

def extract_nonwhite_mask(png_path: str,
                          threshold: int = 250) -> np.ndarray:
    """
    Extract the "non-white" region from a PNG image and generate a binary mask.

    Parameters
    ----------
    png_path : str
        Path to the PNG file to process.
    threshold : int, default 250
        White threshold. A pixel is considered white if R, G, and B ≥ threshold.
        Range 0–255; the higher the value, the stricter the definition. 
        Can be adjusted according to image brightness.
    save_path : str | None
        If provided, saves the mask as a transparent PNG (black background, alpha as mask).

    Returns
    -------
    mask : np.ndarray, dtype=uint8, shape=(H, W)
        Values are 0/1; 1 indicates a non-white pixel.
    """
    img = Image.open(png_path).convert("RGBA").resize((224, 224), Image.LANCZOS)
    arr = np.asarray(img)                 # (H, W, 4) RGBA

    rgb = arr[:, :, :3]
    mask = (rgb < threshold).any(axis=-1).astype(np.uint8)  # 0/1

    return mask

def tolerant_diff_masks(mask1: np.ndarray,
                        mask2: np.ndarray,
                        tolerance_px: int = 3,
                        min_region_area: int | None = 50,
                        show: bool = True,
                        figsize: tuple[int, int] = (12, 4),
                        diff_cmap: str = "Reds") -> np.ndarray:
    """
    Compute “mask₂ - mask₁ (with tolerance for small deviations)” and visualize the result.

    Parameters
    ----------
    mask1, mask2 : np.ndarray (H, W)
        Binary arrays of 0/1 or bool type, must have the same shape.
    tolerance_px : int
        Maximum allowed pixel-level deviation; equivalent to the dilation radius
        (Manhattan distance) applied to mask1. Larger values ⇒ more tolerance,
        fewer extra painted regions remain.
    min_region_area : int | None
        Apply area filtering to the difference result; if None, skip filtering.
        Recommended range: 20–200, adjusted according to image resolution and noise level.
    show : bool
        Whether to use matplotlib to display the original, reconstructed, and difference images.
    figsize : tuple
        Figure size for visualization.
    diff_cmap : str
        Colormap for the difference image.

    Returns
    -------
    extra_mask : np.ndarray, dtype=uint8, shape=(H, W)
        0/1 difference result containing only significant extra-painted regions.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"mask 尺寸不一致: {mask1.shape} vs {mask2.shape}")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    if tolerance_px > 0:
        struct = np.ones((2 * tolerance_px + 1,
                          2 * tolerance_px + 1), dtype=bool)
        m1_dilated = binary_dilation(m1, structure=struct)
    else:
        m1_dilated = m1

    extra = m2 & ~m1_dilated

    if min_region_area is not None and min_region_area > 0:
        labeled, ncomp = label(extra)
        keep = np.zeros_like(extra, dtype=bool)
        for comp_id in range(1, ncomp + 1):
            area = (labeled == comp_id).sum()
            if area >= min_region_area:
                keep |= (labeled == comp_id)
        extra = keep

    extra_mask = extra.astype(np.uint8)    # 0/1

    if show:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(m1, cmap="gray")
        axes[0].set_title("mask₁ Original")
        axes[0].axis("off")

        axes[1].imshow(m2, cmap="gray")
        axes[1].set_title("mask₂ Generated")
        axes[1].axis("off")

        axes[2].imshow(extra_mask, cmap=diff_cmap)
        axes[2].set_title(
            f"Extra Region\n(tol={tolerance_px}px, min_area={min_region_area})"
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return extra_mask

def color_range_rgb(target_rgb, threshold):
    """
    Generate lower and upper bounds in rgb color space based on target RGB value and threshold.
    OpenCV uses rgb, so the order is automatically converted.

    Args:
        target_rgb: tuple(int, int, int), RGB color, e.g. (144, 238, 144)
        threshold: int, range of color channel variation, e.g. 30 means ±30

    Returns:
        lower_rgb: np.array([B, G, R])
        upper_rgb: np.array([B, G, R])
    """
    lower = np.clip(np.array(target_rgb) - threshold, 0, 255)
    upper = np.clip(np.array(target_rgb) + threshold, 0, 255)
    return lower.astype(np.uint8), upper.astype(np.uint8)

def extract_exact_color_mask(
    img_path: str,
    target_rgb: Tuple[int, int, int] = (255, 0, 0),
    tolerance: int = 0,
    resize_to: Optional[Tuple[int, int]] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Extract a pixel mask matching target_rgb exactly (or within a tolerance).

    Parameters
    ----------
    img_path : str
        Path to the input image.
    target_rgb : (R, G, B)
        Target color, default (0, 0, 128).
    tolerance : int, default 0
        Pixel is considered a match if |channel - target| ≤ tol; 0 means exact match.
    resize_to : (W, H) | None
        If specified, resize the image to this size before matching.
    show : bool, default False
        If True, display side-by-side: original / mask / overlay.

    Returns
    -------
    mask : np.ndarray, uint8, shape=(H, W)
        Binary mask with values 0/255; 255 indicates pixels matching target_rgb.
    """ 
    img = Image.open(img_path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)          # (H, W, 3)

    lower, upper = color_range_rgb(target_rgb, tolerance)
    mask = cv2.inRange(arr, lower, upper)
    
    if show:
        overlay = img.copy()
        overlay_arr = np.asarray(overlay).copy()
        overlay_arr[mask == 255] = [255, 0, 0]      # highlight in red
        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [("Original", img),
             ("Mask", Image.fromarray(mask)),
             ("Overlay", Image.fromarray(overlay_arr))]
        ):
            plt.subplot(1, 3, i + 1)
            plt.imshow(im)
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask

def compare_masks_morph(
    mask_ref: np.ndarray,
    mask_test: np.ndarray,
    tol_ref: int = 3,
    tol_tst: int = 3,
    min_region_area: int | None = None,
    min_region_span=20,     
    show: bool = False,
):
    """
    Compare two binary masks with symmetric morphological tolerance.

    Parameters
    ----------
    mask_ref, mask_test : uint8 ndarray (H, W)
        Binary masks with pixel values {0,1} or {0,255}; normalized internally.
    tol : int, default 3
        Tolerance radius (≥0). If tol=0, comparison is strict pixel-by-pixel.
    min_region_area : int | None
        Apply area filtering to missing/extra connected regions. 
        Regions smaller than this threshold are ignored. 
        If None, no filtering is applied.
    show : bool
        If True, plot the difference image and display the number of missing/extra regions in the title.

    Returns
    -------
    dict
        Dictionary with keys: num_missing, num_extra, diff_mask, ref_proc, test_proc
    """
    if mask_ref.shape != mask_test.shape:
        raise ValueError(f"Shape mismatch: {mask_ref.shape} vs {mask_test.shape}")

    ref = mask_ref
    tst = mask_test

    if tol_ref > 0 and tol_tst > 0:
        kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_ref + 1, 2 * tol_ref + 1))
        kernel_tst = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_tst + 1, 2 * tol_tst + 1))
        ref_dil  = cv2.dilate(ref, kernel_ref)
        tst_dil  = cv2.dilate(tst, kernel_tst)
    else:
        ref_dil, tst_dil = ref, tst

    missing = (ref == 1) & (tst_dil == 0)
    extra   = (tst == 1) & (ref_dil == 0)

    def filter_by_geom(bin_mask: np.ndarray,
                   min_area: int | None,
                   min_span: int | None) -> np.ndarray:
        if (min_area is None or min_area <= 0) and \
        (min_span is None or min_span <= 0):
            return bin_mask                         

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask.astype(np.uint8), connectivity=8)

        # OpenCV stats columns
        LEFT, TOP, WIDTH, HEIGHT, AREA = range(5)

        keep = np.zeros_like(bin_mask, dtype=bool)

        for i in range(1, n):
            area  = stats[i, AREA]
            width = stats[i, WIDTH]
            height = stats[i, HEIGHT]
            span = max(width, height)               

            cond_area = (min_area is None or area  >= min_area)
            cond_span = (min_span is None or span  >= min_span)

            if cond_area and cond_span:             
                keep |= (lbl == i)

        return keep

    missing = filter_by_geom(missing, min_region_area, min_region_span)
    extra   = filter_by_geom(missing, min_region_area, min_region_span)

    num_missing = int(missing.sum())
    num_extra   = int(extra.sum())

    diff = np.zeros_like(ref, dtype=np.uint8)
    diff[missing] = 128    # yellow
    diff[extra]   = 255    # red

    if show:
        disp = np.zeros_like(diff, dtype=np.uint8)
        disp[diff == 128] = 1
        disp[diff == 255] = 2
        cmap = ListedColormap(["black", "yellow", "red"])

        plt.figure(figsize=(6, 6))
        plt.imshow(disp, cmap=cmap, vmin=0, vmax=2)
        plt.title(f"diff  (missing={num_missing}, extra={num_extra}, tol={tol_ref}px)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "num_missing": num_missing,
        "num_extra": num_extra,
        "diff_mask": diff,
        "ref_proc":  (ref * 255).astype(np.uint8),
        "test_proc": (tst * 255).astype(np.uint8),
    }

def has_fuzzy_overlap(
    target_mask: np.ndarray,
    extra_path_path: np.ndarray,
    tol: int = 3,
    min_pixels: int = 1
) -> Tuple[bool, int, float]:
    """
    Check whether extra_path_path “loosely” overlaps with target_mask.

    Parameters
    ----------
    target_mask : uint8 ndarray (H, W)
        Reference mask (0/255 or 0/1).
    extra_path_path : uint8 ndarray (H, W)
        Mask to be tested.
    tol : int, default 3
        Dilation radius in pixels. If 0, no dilation is applied.
    min_pixels : int, default 1
        Minimum number of overlapping pixels required to count as overlap.

    Returns
    -------
    has_overlap : bool
        True if the overlap condition is satisfied.
    """
    if target_mask.shape != extra_path_path.shape:
        raise ValueError(f"Shape mismatch: {target_mask.shape} vs {extra_path_path.shape}")

    tgt = (target_mask > 128).astype(np.uint8)
    oth = (extra_path_path   > 128).astype(np.uint8)

    if tol > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        tgt_dil = cv2.dilate(tgt, kernel)
    else:
        tgt_dil = tgt

    inter = (tgt_dil & oth).sum()

    has_overlap = inter >= min_pixels

    return has_overlap, int(inter)

def image_evaluator(output_path, label_path, input_path):
    """
    Evaluate the generated image against the input image and label.

    Parameters:
        output_path (str): Path to the generated image.
        label_path (str): Path to the ground truth image.
        input_path (str): Path to the input image.

    Returns:
        bool: True if the generated image is valid, False otherwise.
    """

    input_mask = extract_nonwhite_mask(input_path)
    gt_mask = extract_nonwhite_mask(label_path)
    ouput_mask = extract_nonwhite_mask(output_path)
    target_mask = extract_exact_color_mask(
                            input_path,
                            target_rgb=(255, 0, 0),
                            tolerance=5,     
                            resize_to=(224, 224),
                            show=False
                        )
    binary_target_mask = (target_mask // 255).astype(np.uint8)

    gt_path = tolerant_diff_masks(
        input_mask, gt_mask,
        tolerance_px=3,        
        min_region_area=100,   
        show=False
    )

    output_path_w_target = tolerant_diff_masks(
        input_mask-binary_target_mask, ouput_mask,
        tolerance_px=3,        
        min_region_area=100,   
        show=False
    )

    output_path = tolerant_diff_masks(
        binary_target_mask, output_path_w_target,
        tolerance_px=3,        
        min_region_area=100,   
        show=False
    )

    result = compare_masks_morph(
        gt_path, output_path,
        tol_ref=5,  
        tol_tst=3,    
        min_region_area=80, 
        min_region_span=10,         
        show=False         
    )

    missing = result["num_missing"]
    extra = result["num_extra"]
    if missing > 150:
        return {"arrived": False, "excess": False, "explore": False}
    elif extra > 80:
        # 检查 extra_path 是否与目标颜色有重叠
        extra_path = (result["diff_mask"] == 255).astype(np.uint8) * 255
        target_mask = extract_exact_color_mask(
                            input_path,
                            target_rgb=(255, 0, 0),
                            tolerance=5,     
                            resize_to=(224, 224),
                            show=False
                        )
        has_overlap, overlap_count = has_fuzzy_overlap(
            target_mask, extra_path,
            tol=5, min_pixels=1
        )
        if has_overlap:
            return {"arrived": True, "excess": True, "explore": False}
        else:
            return {"arrived": True, "excess": False, "explore": True}
    else:
        return {"arrived": True, "excess": False, "explore": False}

def evaluate_layout(output_path, input_path):
    input_wall_mask = extract_exact_color_mask(
                            input_path,
                            target_rgb=(0, 0, 0),
                            tolerance=5,     
                            resize_to=(224, 224),
                            show=False
                        )
    input_wall_mask = (input_wall_mask // 255).astype(np.uint8)
    output_wall_mask = extract_exact_color_mask(
                            output_path,
                            target_rgb=(0, 0, 0),
                            tolerance=5,     
                            resize_to=(224, 224),
                            show=False
                        )
    output_wall_mask = (output_wall_mask // 255).astype(np.uint8)
    result = compare_masks_morph(
        input_wall_mask, output_wall_mask,
        tol_ref=15,  
            tol_tst=3,    
            min_region_area=None,
            min_region_span=None,
        show=False
    )
    missing = result["num_missing"]
    extra = result["num_extra"]
    if missing > 200 :
        return False
    else:
        return True


def extract_action_from_text(text):
    matches = re.findall(r"<actions>(.*?)</actions>", text, re.DOTALL)
    if matches:
        actions_dict = {}
        actions_list = []
        for idx, actions_str in enumerate(matches):
            actions = [action.strip() for action in actions_str.split(',')]
            actions_dict[idx] = actions
            actions_list.extend(actions)
        return actions_list, actions_dict
    else:
        return None, None

def extract_answer_from_text(text):
    match = re.search(r"<answer>(.*?)</answer>", text)
    if match:
        answer_str = match.group(1).lower()
        answer_list = [answer.strip() for answer in answer_str.split(',')]
        return answer_list
    else:
        return None

def text_evaluator(output, label):
    """
    Compute the prefix accuracy of a predicted action sequence compared to the label.

    Parameters:
        output (str): Raw output string containing actions (e.g., "<go forward><turn left>").
        label (List[str]): Ground truth list of actions.
        endpoint (str): An optional string used by extract_action to parse actions.

    Returns:
        float: Accuracy = number of correct actions from the start / total predicted actions.
               Returns 0.0 if predicted sequence is empty.
        int: Length of the predicted action sequence.
    """
    entire_action_sequence, action_sequences = extract_action_from_text(output)
    
    answer_sequence = extract_answer_from_text(output)
        
    if not entire_action_sequence:
        return {
        "accuracy": 0.0,
        "state_accuracy": 0.0,
        "next_state_len": 0,
        "correct": 0,
        "next_action_sequence": None,
        "entire_action_sequence": None,
        "answer_sequence": None
    }
    
    correct = 0
    for pred_action, true_action in zip(entire_action_sequence, label):
        if pred_action == true_action:
            correct += 1
        else:
            break
    state_correct = 0
    next_state_len = len(action_sequences[0])
    for pred_action, true_action in zip(action_sequences[0], label):
        if pred_action == true_action:
            state_correct += 1
        else:
            break

    return {
        "accuracy": correct / len(label),
        "state_accuracy": state_correct / next_state_len,
        "next_state_len": next_state_len,
        "correct": correct,
        "next_action_sequence": action_sequences[0],
        "entire_action_sequence": action_sequences,
        "answer_sequence": answer_sequence
    }

