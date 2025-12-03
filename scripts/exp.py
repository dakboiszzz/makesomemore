import torch

# 1. Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Move data to the GPU
# The .to(device) moves the data from RAM to VRAM (Video Memory)
x = torch.rand(5000, 10000).to(device)
y = torch.rand(10000, 5000).to(device)

# 3. Perform calculation
# Since x and y are on GPU, the math happens on the RTX 3050
z = torch.matmul(x, y)

# 4. (Optional) Move result back to CPU if you need to print/save it
z = z.cpu()
print("Done")
