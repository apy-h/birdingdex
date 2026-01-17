"""
Test script to debug the download_via_kaggle function.
Tests if the Kaggle API token is working correctly.
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Check which format we have and convert if needed
kaggle_api_token = os.getenv('KAGGLE_API_TOKEN')
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

if kaggle_api_token:
    # Newer token format (KGAT_...) - needs to be set as KAGGLE_KEY with __token__ username
    os.environ['KAGGLE_USERNAME'] = '__token__'
    os.environ['KAGGLE_KEY'] = kaggle_api_token
    print(f"Converted KAGGLE_API_TOKEN to KAGGLE_USERNAME/__token__ format")
elif kaggle_username and kaggle_key:
    print(f"Using classic KAGGLE_USERNAME and KAGGLE_KEY format")
else:
    print("ERROR: No Kaggle credentials found")
    sys.exit(1)


def test_kaggle_download():
    """Test the Kaggle download function in isolation."""

    print(f"\n{'='*60}")
    print("KAGGLE DOWNLOAD TEST")
    print(f"{'='*60}")

    # Check if KAGGLE_API_TOKEN is set
    kaggle_token = os.getenv('KAGGLE_API_TOKEN')
    print(f"\nEnvironment Check:")
    print(f"  KAGGLE_API_TOKEN set: {bool(kaggle_token)}")
    if kaggle_token:
        print(f"  Token preview: {kaggle_token[:10]}...{kaggle_token[-10:]}")

    # Try to import kaggle
    print(f"\nImport Check:")
    try:
        import kaggle
        print(f"  ✓ Kaggle module imported successfully")
        print(f"  Kaggle version: {kaggle.__version__ if hasattr(kaggle, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"  ✗ Failed to import kaggle: {e}")
        print(f"  Install with: pip install kaggle")
        return False

    # Check Kaggle credentials
    print(f"\nCredentials Check:")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()

        # Check if credentials file exists
        kaggle_config_dir = os.path.expanduser('~/.kaggle')
        kaggle_json = os.path.join(kaggle_config_dir, 'kaggle.json')
        env_token_set = bool(os.getenv('KAGGLE_API_TOKEN'))

        print(f"  Kaggle config dir: {kaggle_config_dir}")
        print(f"  kaggle.json exists: {os.path.exists(kaggle_json)}")
        print(f"  KAGGLE_API_TOKEN env var set: {env_token_set}")

        # Try to authenticate
        api.authenticate()
        print(f"  ✓ Authentication successful!")

    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print(f"\n  How to fix:")
        print(f"    1. Go to: https://www.kaggle.com/settings/account")
        print(f"    2. Click 'Create New Token' (downloads kaggle.json)")
        print(f"    3. Place kaggle.json in: {kaggle_config_dir}")
        print(f"    OR set KAGGLE_API_TOKEN environment variable with the token from step 2")
        return False

    # Test actual dataset download
    print(f"\nDataset Download Test:")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            kaggle_dataset = 'wenewone/cub2002011'
            print(f"  Attempting to download: {kaggle_dataset}")
            print(f"  Destination: {temp_dir}")

            import time
            start = time.time()
            api.dataset_download_files(
                kaggle_dataset,
                path=temp_dir,
                unzip=False,  # Don't unzip, just download
                quiet=False
            )
            elapsed = time.time() - start

            # Check what was downloaded
            files = os.listdir(temp_dir)
            print(f"  ✓ Download successful! ({elapsed:.1f}s)")
            print(f"  Downloaded files: {files}")

            return True

        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            print(f"\n  Possible issues:")
            print(f"    - Invalid or expired token")
            print(f"    - Kaggle API rate limit exceeded")
            print(f"    - Dataset not accessible or moved")
            print(f"    - Network/firewall issues")
            return False

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    success = test_kaggle_download()
    exit(0 if success else 1)
