"""
Test script to verify checkpoint cleanup functionality on existing directories.
Runs the cleanup logic on your actual models/ directory.
"""

import os
import shutil
import sys


def test_checkpoint_cleanup(output_dir=None):
    """Test checkpoint cleanup logic on existing directories."""

    # Default to backend/models if not specified
    if output_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, 'models')

    print(f"\n{'='*60}")
    print("CHECKPOINT CLEANUP TEST")
    print(f"{'='*60}")
    print(f"\nTarget directory: {output_dir}")

    if not os.path.isdir(output_dir):
        print(f"✗ Directory does not exist: {output_dir}")
        return False

    checkpoints_root = os.path.join(output_dir, 'checkpoints')

    print(f"\nBefore cleanup:")
    print(f"  output_dir contents: {os.listdir(output_dir)}")
    print(f"  checkpoints/ exists: {os.path.isdir(checkpoints_root)}")
    if os.path.isdir(checkpoints_root):
        checkpoints_in_root = os.listdir(checkpoints_root)
        print(f"  checkpoints/ contents ({len(checkpoints_in_root)} items): {checkpoints_in_root}")

    # Run the cleanup logic (same as in train_model.py)
    print(f"\nRunning cleanup...")
    checkpoint_count = 0

    if os.path.isdir(checkpoints_root):
        for item in os.listdir(checkpoints_root):
            item_path = os.path.join(checkpoints_root, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                try:
                    shutil.rmtree(item_path)
                    checkpoint_count += 1
                    print(f"  ✓ Deleted {item}")
                except Exception as e:
                    print(f"  ✗ Warning: Failed to delete {item}: {e}")

        # Remove the empty checkpoints root if all subfolders were deleted
        try:
            if not os.listdir(checkpoints_root):
                shutil.rmtree(checkpoints_root)
                print(f"  ✓ Deleted empty checkpoints/ directory")
        except Exception:
            pass

    if checkpoint_count > 0:
        print(f"\n✓ Deleted {checkpoint_count} checkpoint director{'y' if checkpoint_count == 1 else 'ies'}")
    else:
        print(f"\n  No checkpoint directories found")

    print(f"\nAfter cleanup:")
    print(f"  output_dir contents: {os.listdir(output_dir)}")
    print(f"  checkpoints/ exists: {os.path.isdir(checkpoints_root)}")

    print(f"\n{'='*60}\n")
    return True


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_checkpoint_cleanup(output_dir)
