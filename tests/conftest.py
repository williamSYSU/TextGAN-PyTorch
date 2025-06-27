import pytest
import tempfile
import shutil
import os
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    class MockConfig:
        def __init__(self):
            self.vocab_size = 5000
            self.gen_embed_dim = 32
            self.gen_hidden_dim = 32
            self.dis_embed_dim = 32
            self.dis_hidden_dim = 32
            self.max_seq_len = 20
            self.batch_size = 64
            self.num_rep = 64
            self.gen_lr = 0.01
            self.dis_lr = 0.01
            self.update_rate = 0.8
            self.temperature = 1.0
            self.training_data = "oracle"
            self.dataset = "oracle"
            self.model_type = "vanilla"
            self.gen_init = "normal"
            self.dis_init = "uniform"
            self.eval_type = "standard"
            self.tips = 2000
            self.temp_adpt = "exp"
            self.oracle_pretrain = True
            self.dis_pretrain = True
            self.adv_g_step = 1
            self.rollout_num = 16
            self.gen_pretrain_steps = 120
            self.dis_pretrain_steps = 50
            self.log_file = "log/test_log.txt"
            self.save_root = "save/test/"
            self.signal_file = "run_signal.txt"
            self.tips = 2000
            
    return MockConfig()


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    return torch.randn(64, 20)  # batch_size x seq_len


@pytest.fixture
def sample_numpy_array():
    """Provide a sample numpy array for testing."""
    return np.random.randn(64, 20)


@pytest.fixture
def mock_data_loader():
    """Provide a mock data loader for testing."""
    class MockDataLoader:
        def __init__(self, batch_size=64):
            self.batch_size = batch_size
            self.num_batch = 100
            
        def __iter__(self):
            for _ in range(self.num_batch):
                yield torch.randint(0, 5000, (self.batch_size, 20))
                
        def __len__(self):
            return self.num_batch
            
        def reset_pointer(self):
            pass
            
    return MockDataLoader()


@pytest.fixture
def mock_generator():
    """Provide a mock generator model for testing."""
    class MockGenerator(torch.nn.Module):
        def __init__(self, vocab_size=5000, embed_dim=32, hidden_dim=32):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            
        def forward(self, x):
            batch_size = x.size(0) if x.dim() > 0 else 1
            return torch.randn(batch_size, 20, self.vocab_size)
            
        def sample(self, batch_size, seq_len):
            return torch.randint(0, self.vocab_size, (batch_size, seq_len))
            
    return MockGenerator()


@pytest.fixture
def mock_discriminator():
    """Provide a mock discriminator model for testing."""
    class MockDiscriminator(torch.nn.Module):
        def __init__(self, vocab_size=5000, embed_dim=32, hidden_dim=32):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            
        def forward(self, x):
            batch_size = x.size(0)
            return torch.rand(batch_size, 1)
            
    return MockDiscriminator()


@pytest.fixture
def clean_logs(temp_dir):
    """Ensure log directories are clean for testing."""
    log_dir = temp_dir / "log"
    save_dir = temp_dir / "save"
    log_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)
    yield log_dir, save_dir


@pytest.fixture
def mock_oracle_data(temp_dir):
    """Create mock oracle data for testing."""
    oracle_file = temp_dir / "oracle.txt"
    with open(oracle_file, 'w') as f:
        for _ in range(10000):
            # Generate random sequences of integers
            seq = ' '.join(str(np.random.randint(0, 5000)) for _ in range(20))
            f.write(seq + '\n')
    return oracle_file


@pytest.fixture(autouse=True)
def setup_random_seeds():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")