import torch

# Setup PyTorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

data_path = "data"
img_size = (128, 64)
# Number of cells per axis in each layer
layers = [1, 2, 4]
