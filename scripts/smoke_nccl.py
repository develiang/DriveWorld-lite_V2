from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        rank = dist.get_rank()
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)
        print(
            f"rank={rank} device={device} nccl_all_reduce={float(value):.1f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
