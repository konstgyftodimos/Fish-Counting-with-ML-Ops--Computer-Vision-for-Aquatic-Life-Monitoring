# Libraries
import torch

def mps_check():
    """
    Check MPU GPU availability
    """
    print(f"Pytorch version: {torch.__version__}")
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS (Metal Performance Shader) available? {torch.backends.mps.is_available()}")

if __name__ == '__main__':
    mps_check()