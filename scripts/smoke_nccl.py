from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description="NCCL collective smoke test")
    parser.add_argument("--numel", type=int, default=1)
    parser.add_argument("--collective", choices=["all-reduce", "broadcast"], default="all-reduce")
    args = parser.parse_args()
    if args.numel <= 0:
        raise SystemExit("--numel must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        rank = dist.get_rank()
        value = torch.full((args.numel,), float(rank + 1), device=device)
        if args.collective == "all-reduce":
            dist.all_reduce(value)
            expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        else:
            dist.broadcast(value, src=0)
            expected = 1.0
        torch.cuda.synchronize(device)
        if not bool(torch.all(value == expected)):
            raise RuntimeError(
                f"NCCL {args.collective} mismatch on rank {rank}: "
                f"first={float(value[0])} expected={expected}"
            )
        print(
            f"rank={rank} device={device} collective={args.collective} "
            f"numel={args.numel} result={float(value[0]):.1f}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
