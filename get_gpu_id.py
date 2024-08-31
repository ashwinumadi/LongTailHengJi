import torch
print("######")
print(torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")