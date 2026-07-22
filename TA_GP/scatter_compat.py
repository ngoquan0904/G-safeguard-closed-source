"""
Thay thế `torch_scatter.scatter_mean`.

Lý do: torch_scatter là extension phải compile theo đúng phiên bản torch, và
không có wheel dựng sẵn cho torch mới (2.13 CPU) -> pip build fail. Trong khi
torch_geometric đã có sẵn `utils.scatter` làm đúng việc đó bằng
torch.scatter_reduce native, không cần compile.

Chữ ký giữ y hệt torch_scatter.scatter_mean nên call site không phải đổi.
"""
from torch_geometric.utils import scatter as _scatter


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    if out is not None:                       # không call site nào dùng, chặn cho chắc
        raise NotImplementedError("scatter_compat.scatter_mean không hỗ trợ out=")
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
