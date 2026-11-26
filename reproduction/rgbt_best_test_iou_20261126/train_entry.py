"""Launch the unchanged trainer with the missing PyTorch utility backported."""
import sys
from pathlib import Path

from torch_compat import install

install()
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import train

if __name__ == '__main__':
    train.main()
