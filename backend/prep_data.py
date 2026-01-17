"""Data preparation utilities for the CUB-200-2011 dataset.

This module handles downloading, caching (raw and resized), balancing,
and train/test splitting for the bird classifier pipeline.
"""

import os
import time
import pickle
import shutil
import tarfile
import urllib.request
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def organize_extracted_dataset(extracted_parent_path: str, dataset_path: str) -> None:
    """Move extracted dataset into the standard location and clean up temp dir."""
    extracted_dirs = [d for d in os.listdir(extracted_parent_path)
                      if os.path.isdir(os.path.join(extracted_parent_path, d))]

    if extracted_dirs:
        src = os.path.join(extracted_parent_path, extracted_dirs[0])
        shutil.move(src, dataset_path)
        print(f"  ✓ Dataset moved to: {dataset_path}")
    else:
        if os.path.exists(extracted_parent_path):
            os.rename(extracted_parent_path, dataset_path)
            print(f"  ✓ Dataset ready at: {dataset_path}")

    if os.path.exists(extracted_parent_path):
        shutil.rmtree(extracted_parent_path)


def download_via_kaggle(data_dir: str, dataset_path: str) -> bool:
    """Attempt to download the dataset via Kaggle.

    Returns True on success, False otherwise.
    """
    print("\nMethod 1: Trying Kaggle API...")
    kaggle_path = os.path.join(data_dir, 'kaggle_download')
    os.makedirs(kaggle_path, exist_ok=True)

    kaggle_dataset = 'wenewone/cub2002011'
    try:
        import kaggle  # Imported lazily to avoid hard dependency when not needed

        print(f"  Downloading from Kaggle: {kaggle_dataset}")
        start_time = time.time()
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=kaggle_path,
            unzip=True,
            quiet=False
        )
        elapsed = time.time() - start_time
        print(f"  ✓ Successfully downloaded from Kaggle! ({elapsed:.1f}s)")

        organize_extracted_dataset(kaggle_path, dataset_path)
        return True
    except Exception as e:
        print(f"  ✗ Kaggle download failed: {e}")
        print("  If needed, configure Kaggle API credentials:")
        print("    https://www.kaggle.com/settings/account -> Create New Token")
        return False


def download_via_direct(data_dir: str, dataset_path: str) -> None:
    """Download the dataset via direct link from Caltech servers."""
    print("\nMethod 2: Direct download from Caltech...")
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tgz_path = os.path.join(data_dir, 'CUB_200_2011.tgz')

    print(f"  Downloading from: {url}")
    print("  Size: ~1.1 GB - This may take several minutes...")

    start_time = time.time()
    urllib.request.urlretrieve(url, tgz_path)
    elapsed = time.time() - start_time
    print(f"  ✓ Download complete! ({elapsed:.1f}s)")

    print("  Extracting archive...")
    extract_start = time.time()
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    extract_elapsed = time.time() - extract_start
    print(f"  ✓ Extraction complete! ({extract_elapsed:.1f}s)")

    organize_extracted_dataset(data_dir, dataset_path)

    if os.path.exists(tgz_path):
        os.remove(tgz_path)


def load_raw_images_from_disk(dataset_path: str) -> Tuple[List[Image.Image], np.ndarray, List[str]]:
    """Load raw images and labels from the dataset directory."""
    images_path = os.path.join(dataset_path, 'images')
    images_file = os.path.join(dataset_path, 'images.txt')
    labels_file = os.path.join(dataset_path, 'image_class_labels.txt')
    classes_file = os.path.join(dataset_path, 'classes.txt')

    class_names: List[str] = []
    with open(classes_file, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            class_names.append(class_name)

    print(f"  Found {len(class_names)} bird species")

    image_paths = {}
    with open(images_file, 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split(' ', 1)
            image_paths[img_id] = img_path

    image_labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            img_id, class_id = line.strip().split(' ')
            image_labels[img_id] = int(class_id) - 1

    raw_images: List[Image.Image] = []
    labels = []

    print("  Loading raw images from disk...")
    for img_id in tqdm(sorted(image_paths.keys()), desc="Loading images"):
        img_path = os.path.join(images_path, image_paths[img_id])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                raw_images.append(img)
                labels.append(image_labels[img_id])
            except Exception as e:
                print(f"    Warning: Failed to load {img_path}: {e}")
                continue

    labels_array = np.array(labels)
    print(f"  ✓ Loaded {len(raw_images)} raw images")

    return raw_images, labels_array, class_names


def resize_images(images: List[Image.Image], target_size: int = 224) -> List[Image.Image]:
    """Resize images to the target size."""
    print(f"  Resizing images to {target_size}x{target_size}...")
    resized_images: List[Image.Image] = []
    for img in tqdm(images, desc="Resizing images"):
        resized_images.append(img.resize((target_size, target_size)))
    print(f"  ✓ Resized {len(resized_images)} images")
    return resized_images


def balance_dataset(images: List[Image.Image], labels: np.ndarray, class_names: List[str],
                    max_samples_per_class: int) -> Tuple[List[Image.Image], np.ndarray]:
    """Balance dataset by limiting samples per class."""
    print(f"  Balancing dataset (max {max_samples_per_class} samples per class)...")
    balanced_indices = []

    for class_idx in range(len(class_names)):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            selected = np.random.choice(
                class_indices,
                size=min(len(class_indices), max_samples_per_class),
                replace=False
            )
            balanced_indices.extend(selected)

    balanced_indices = np.array(balanced_indices)
    balanced_images = [images[i] for i in balanced_indices]
    balanced_labels = labels[balanced_indices]

    print(f"  ✓ Balanced dataset: {len(balanced_images)} images across {len(class_names)} classes")
    return balanced_images, balanced_labels


def split_train_test(images: List[Image.Image], labels: np.ndarray,
                     test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    print(f"  Splitting into train/test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    print(f"  ✓ Train set: {len(X_train)} images")
    print(f"  ✓ Test set: {len(X_test)} images")
    return X_train, X_test, y_train, y_test


def save_sample_image(all_images: List[Image.Image], all_labels: np.ndarray,
                      class_names: List[str], data_dir: str) -> None:
    """Save a sample image for quick verification."""
    if not all_images:
        return
    sample_idx = 0
    sample_path = os.path.join(data_dir, 'sample_image.png')
    all_images[sample_idx].save(sample_path)
    print(f"  ✓ Sample image saved to: {sample_path}")
    print(f"    Sample label: {class_names[all_labels[sample_idx]]}")


def load_resized_cache(cache_path: str) -> Optional[Tuple[List[Image.Image], np.ndarray, List[str]]]:
    """Load resized cache if present."""
    if not os.path.exists(cache_path):
        return None
    print("✓ Found resized images cache")
    print(f"  Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
        return cache_data['images'], cache_data['labels'], cache_data['class_names']


def load_raw_cache(cache_path: str) -> Optional[Tuple[List[Image.Image], np.ndarray, List[str]]]:
    """Load raw cache if present."""
    if not os.path.exists(cache_path):
        return None
    print("✓ Found raw images cache")
    print(f"  Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
        return cache_data['images'], cache_data['labels'], cache_data['class_names']


def save_raw_cache(cache_path: str, images: List[Image.Image], labels: np.ndarray,
                   class_names: List[str]) -> None:
    print("  Saving raw images to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump({'images': images, 'labels': labels, 'class_names': class_names}, f)
    print(f"  ✓ Raw cache saved to: {cache_path}")


def save_resized_cache(cache_path: str, images: List[Image.Image], labels: np.ndarray,
                       class_names: List[str]) -> None:
    print("  Saving resized images to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump({'images': images, 'labels': labels, 'class_names': class_names}, f)
    print(f"  ✓ Resized cache saved to: {cache_path}")


def load_cub_dataset(data_dir: str, max_samples_per_class: int, target_size: int = 224):
    """Load dataset following the strict step-by-step flow with two-tier caching."""
    print("\n" + "=" * 60)
    print("LOADING CUB-200-2011 DATASET")
    print("=" * 60)

    load_start = time.time()
    os.makedirs(data_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, 'CUB_200_2011')
    raw_cache_path = os.path.join(data_dir, 'cub_cache_raw.pkl')
    resized_cache_path = os.path.join(data_dir, f'cub_cache_resized_{target_size}.pkl')

    # Step 1: Check if resized images cache exists (Cache 2)
    resized_cache = load_resized_cache(resized_cache_path)
    if resized_cache:
        all_images, all_labels, class_names = resized_cache
    else:
        # Step 2: Check if raw images cache exists (Cache 1)
        raw_cache = load_raw_cache(raw_cache_path)
        if raw_cache:
            raw_images, all_labels, class_names = raw_cache
        else:
            # Step 3: Try to download via Kaggle
            if not os.path.exists(dataset_path):
                success = download_via_kaggle(data_dir, dataset_path)
                # Step 4: Try to download via direct link if Kaggle failed
                if not success and not os.path.exists(dataset_path):
                    download_via_direct(data_dir, dataset_path)
            else:
                print(f"Dataset already present at {dataset_path}, skipping download")

            # Step 5: Save cache of raw images (after loading from disk)
            raw_images, all_labels, class_names = load_raw_images_from_disk(dataset_path)
            save_raw_cache(raw_cache_path, raw_images, all_labels, class_names)

        # Step 6: Resize images
        all_images = resize_images(raw_images, target_size)
        # Step 7: Save cache of resized images
        save_resized_cache(resized_cache_path, all_images, all_labels, class_names)

    print(f"\n  Total images: {len(all_images)}")
    print(f"  Total classes: {len(class_names)}")

    # Save sample image for verification
    save_sample_image(all_images, all_labels, class_names, data_dir)

    # Step 8: Balance dataset
    balanced_images, balanced_labels = balance_dataset(
        all_images, all_labels, class_names, max_samples_per_class
    )

    # Step 9: Split into train/test sets
    X_train, X_test, y_train, y_test = split_train_test(
        balanced_images, balanced_labels
    )

    load_elapsed = time.time() - load_start
    print(f"\n✓ Dataset loading complete! ({load_elapsed:.1f}s)")

    return X_train, y_train, X_test, y_test, class_names


def download_and_prepare_dataset(data_dir: str = None, max_samples_per_class: int = 100,
                                 image_size: int = 224):
    """Public entry point to download (if needed), cache, balance, and split the dataset."""
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'dataset')

    return load_cub_dataset(data_dir, max_samples_per_class, image_size)
