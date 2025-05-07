import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check the CUDA version PyTorch was built with
print("Built with CUDA version:", torch.version.cuda)

# Check the actual GPU device name
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

