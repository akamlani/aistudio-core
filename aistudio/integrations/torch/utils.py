import torch
import gc

# images usally stored as [n, h, w, ch]
# pytorch format: [ch, h, w] -> e.g., tensor.permute(2, 0, 1)
trsfrm_to_tensor        = lambda x: torch.as_tensor(x, dtype=torch.float32)
trsfrm_tensor_resize    = lambda x, dim: x.reshape(*dim)
trsrm_tensor_format     = lambda x, dim: x.permute(*dim)
trsfrm_tensor_to_device = lambda x, device: x.to(device)

get_type_from_tensor    = lambda x: x.type()
get_shape_from_tensor   = lambda x: x.shape
get_value_from_tensor   = lambda x: x.item()
get_device_from_tensor  = lambda x: x.device

get_device              = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_cuda_device_count   = lambda: torch.cuda.device_count()
get_cuda_devices        = lambda: [i for i in range(torch.cuda.device_count())]
get_cuda_device_name    = lambda: torch.cuda.get_device_name(get_device())
get_is_gpu_avail        = lambda: torch.cuda.is_available()
get_max_memory_used     = lambda: torch.cuda.max_memory_allocated(device="cuda")

trsfrm_numpy_to_tensor  = lambda x: torch.from_numpy(x)
trsfrm_to_tensor        = lambda x: torch.tensor(x)
trsfrm_tensor_dtype     = lambda x, dtype: x.to(dtype)      #x.to(torch.float32)

def get_tensor_properties(inp: torch.Tensor) -> dict:
    return dict(
        type    = inp.dtype,
        shape   = inp.shape,
        rank    = inp.ndim,     # equivalent to len(inp.shape)
        device  = inp.device,
        value   = inp.item()
    )

def get_num_model_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_in_bytes(model) -> float:
    # multiply by 1e-6 for mb
    return sum(p.element_size()*p.numel() for p in model.parameters())

def get_tensor_size_in_bytes(inp: torch.Tensor) -> int:
    return inp.numel() * inp.element_size()

def torch_get_device_properties() -> dict:
    return dict(
        version     = torch.__version__,
        device      = get_device(),
        is_cuda     = get_is_gpu_avail(),
        cuda_count  = get_cuda_device_count(),
        gpu         = torch.cuda.get_device_name(get_device())
    )

def torch_seed_init(seed:int=42) -> None:
    """seeds torch environment and device as applicable

    Args:
        seed (int, optional): seed to configure. Defaults to 42.
    """
    torch.manual_seed(seed)
    if is_gpu_avail():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = True

def torch_memory_clean() -> None:
    """cleans up memory from GPU if available"""
    if is_gpu_avail():
        gc.collect()
        torch.cuda.empty_cache()
