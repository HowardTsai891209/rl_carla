import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    gpu_available = torch.cuda.is_available()
    print(f"GPU available: {gpu_available}")

    if gpu_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected by PyTorch.")

if __name__ == '__main__':
    check_gpu()
