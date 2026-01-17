"""
Test script to verify that get_model_metrics() correctly finds metrics files.
Tests the path lookup logic for both timestamped and non-timestamped models.

This is a lightweight test that simulates the path lookup logic without loading heavy dependencies.
"""

import os
import sys
import json
import tempfile
import shutil


def simulate_get_model_metrics(model_path):
    """
    Simulate the get_model_metrics logic from ml_service.py without importing heavy dependencies.
    This matches the logic in BirdClassifier.get_model_metrics()
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary containing model metrics or default values
    """
    # Look for metrics file in the timestamped model directory (one level up from model_path)
    # Structure: models/bird_classifier_YYYY-MM-DD_HH-MM-SS/model_metrics.json
    #           models/bird_classifier_YYYY-MM-DD_HH-MM-SS/bird_classifier/ <- model_path
    metrics_path = os.path.join(os.path.dirname(model_path), 'model_metrics.json')

    # Fallback to old location for backward compatibility (models/model_metrics.json)
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'model_metrics.json')

    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            # Return default metrics if file not found
            return {
                'model_name': 'Demo Model',
                'num_classes': 2,
                'status': 'Not trained - using demo model',
                'message': 'Run train_model.py to fine-tune the model',
                'hyperparameters': {
                    'num_epochs': 'N/A',
                    'batch_size': 'N/A',
                    'learning_rate': 'N/A',
                },
                'results': {
                    'test_accuracy': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'test_f1': 0.0,
                }
            }
    except Exception as e:
        return {
            'error': str(e),
            'status': 'Error loading metrics'
        }


def create_test_structure(base_dir, with_timestamp=True):
    """
    Create test directory structure with mock model and metrics files.
    
    Args:
        base_dir: Base directory for test structure
        with_timestamp: If True, create timestamped structure; if False, create legacy structure
    
    Returns:
        Tuple of (model_path, expected_metrics_path)
    """
    if with_timestamp:
        # New structure: models/bird_classifier_2026-01-17_12-30-45/
        timestamp_dir = os.path.join(base_dir, 'models', 'bird_classifier_2026-01-17_12-30-45')
        model_dir = os.path.join(timestamp_dir, 'bird_classifier')
        metrics_path = os.path.join(timestamp_dir, 'model_metrics.json')
    else:
        # Old structure: models/bird_classifier/
        model_dir = os.path.join(base_dir, 'models', 'bird_classifier')
        metrics_path = os.path.join(base_dir, 'models', 'model_metrics.json')
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Create mock model files
    config = {
        "id2label": {"0": "Test Bird 1", "1": "Test Bird 2"},
        "label2id": {"Test Bird 1": 0, "Test Bird 2": 1}
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    # Create mock metrics file
    metrics = {
        'model_name': 'test-model',
        'num_classes': 2,
        'hyperparameters': {
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
        },
        'results': {
            'test_accuracy': 0.95,
            'test_precision': 0.94,
            'test_recall': 0.93,
            'test_f1': 0.94,
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return model_dir, metrics_path


def test_timestamped_model_metrics():
    """Test that metrics are found for timestamped models."""
    print(f"\n{'='*60}")
    print("TEST: Timestamped Model Metrics Lookup")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        model_path, expected_metrics_path = create_test_structure(temp_dir, with_timestamp=True)
        
        print(f"\nTest Structure:")
        print(f"  Model path: {model_path}")
        print(f"  Expected metrics path: {expected_metrics_path}")
        print(f"  Metrics file exists: {os.path.exists(expected_metrics_path)}")
        
        # Test the simulated get_model_metrics
        print(f"\nTesting get_model_metrics() logic...")
        metrics = simulate_get_model_metrics(model_path)
        
        # Check if metrics were loaded successfully
        if 'error' in metrics:
            print(f"  ✗ FAILED: Got error in metrics: {metrics['error']}")
            return False
        
        if 'status' in metrics and 'Not trained' in metrics['status']:
            print(f"  ✗ FAILED: Metrics file not found (got default metrics)")
            return False
        
        # Verify expected fields
        expected_accuracy = 0.95
        actual_accuracy = metrics.get('results', {}).get('test_accuracy', 0.0)
        
        if actual_accuracy == expected_accuracy:
            print(f"  ✓ SUCCESS: Metrics loaded correctly")
            print(f"    Test accuracy: {actual_accuracy}")
            print(f"    Model name: {metrics.get('model_name')}")
            return True
        else:
            print(f"  ✗ FAILED: Unexpected accuracy value")
            print(f"    Expected: {expected_accuracy}")
            print(f"    Actual: {actual_accuracy}")
            return False


def test_legacy_model_metrics():
    """Test that metrics are found for non-timestamped (legacy) models."""
    print(f"\n{'='*60}")
    print("TEST: Legacy Model Metrics Lookup (Backward Compatibility)")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        model_path, expected_metrics_path = create_test_structure(temp_dir, with_timestamp=False)
        
        print(f"\nTest Structure:")
        print(f"  Model path: {model_path}")
        print(f"  Expected metrics path: {expected_metrics_path}")
        print(f"  Metrics file exists: {os.path.exists(expected_metrics_path)}")
        
        # Test the simulated get_model_metrics
        print(f"\nTesting get_model_metrics() logic...")
        metrics = simulate_get_model_metrics(model_path)
        
        # Check if metrics were loaded successfully
        if 'error' in metrics:
            print(f"  ✗ FAILED: Got error in metrics: {metrics['error']}")
            return False
        
        if 'status' in metrics and 'Not trained' in metrics['status']:
            print(f"  ✗ FAILED: Metrics file not found (got default metrics)")
            return False
        
        # Verify expected fields
        expected_accuracy = 0.95
        actual_accuracy = metrics.get('results', {}).get('test_accuracy', 0.0)
        
        if actual_accuracy == expected_accuracy:
            print(f"  ✓ SUCCESS: Metrics loaded correctly")
            print(f"    Test accuracy: {actual_accuracy}")
            print(f"    Model name: {metrics.get('model_name')}")
            return True
        else:
            print(f"  ✗ FAILED: Unexpected accuracy value")
            print(f"    Expected: {expected_accuracy}")
            print(f"    Actual: {actual_accuracy}")
            return False


def test_missing_metrics():
    """Test that default metrics are returned when no metrics file exists."""
    print(f"\n{'='*60}")
    print("TEST: Missing Metrics File (Default Fallback)")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal structure without metrics file
        model_dir = os.path.join(temp_dir, 'models', 'bird_classifier')
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nTest Structure:")
        print(f"  Model path: {model_dir}")
        print(f"  Metrics file created: False")
        
        # Test the simulated get_model_metrics
        print(f"\nTesting get_model_metrics() logic...")
        metrics = simulate_get_model_metrics(model_dir)
        
        # Should get default metrics
        if 'status' in metrics and 'Not trained' in metrics['status']:
            print(f"  ✓ SUCCESS: Default metrics returned as expected")
            print(f"    Status: {metrics['status']}")
            return True
        else:
            print(f"  ✗ FAILED: Expected default metrics, got: {metrics}")
            return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL METRICS PATH LOOKUP TESTS")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Timestamped Model", test_timestamped_model_metrics()))
    results.append(("Legacy Model", test_legacy_model_metrics()))
    results.append(("Missing Metrics", test_missing_metrics()))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    exit(0 if passed == total else 1)
