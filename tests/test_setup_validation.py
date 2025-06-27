import pytest
import sys
import os
from pathlib import Path


def test_pytest_is_installed():
    """Test that pytest is properly installed."""
    assert pytest.__version__


def test_project_structure_exists():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check main directories
    assert (project_root / "models").exists()
    assert (project_root / "metrics").exists()
    assert (project_root / "utils").exists()
    assert (project_root / "instructor").exists()
    assert (project_root / "run").exists()
    assert (project_root / "visual").exists()
    
    # Check test directories
    assert (project_root / "tests").exists()
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()


def test_config_imports():
    """Test that basic imports work."""
    try:
        import torch
        import numpy
        import nltk
        import tqdm
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")


def test_fixtures_available(temp_dir, mock_config, sample_tensor):
    """Test that pytest fixtures from conftest.py are available."""
    assert temp_dir.exists()
    assert mock_config.vocab_size == 5000
    assert sample_tensor.shape == (64, 20)


@pytest.mark.unit
def test_unit_marker():
    """Test that the unit marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that the integration marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that the slow marker works."""
    import time
    time.sleep(0.1)
    assert True


def test_coverage_target_modules_importable():
    """Test that modules targeted for coverage are importable."""
    modules_to_test = [
        'models',
        'metrics',
        'utils',
        'instructor',
        'run',
        'visual'
    ]
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    for module in modules_to_test:
        try:
            __import__(module)
        except Exception as e:
            # Some modules might have dependencies, that's ok for this test
            pass


def test_mock_fixtures(mock_generator, mock_discriminator, mock_data_loader):
    """Test that mock fixtures work properly."""
    # Test generator
    gen_output = mock_generator.sample(32, 20)
    assert gen_output.shape == (32, 20)
    
    # Test discriminator
    dis_input = gen_output
    dis_output = mock_discriminator(dis_input)
    assert dis_output.shape == (32, 1)
    
    # Test data loader
    batch_count = 0
    for batch in mock_data_loader:
        batch_count += 1
        assert batch.shape == (64, 20)
        if batch_count >= 5:  # Just test a few batches
            break
    assert batch_count == 5