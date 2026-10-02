import json
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vector_quantize_pytorch.lookup_free_quantization import is_distributed, maybe_distributed_mean


def run_rank(rank, rendezvous, directory, warm_before_init):
    if warm_before_init:
        if is_distributed():
            raise AssertionError("unexpected initialized process group")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        value = torch.tensor([1. + 2 * rank, 2. + 2 * rank], requires_grad=True)
        mean = maybe_distributed_mean(value)
        mean.sum().backward()
        with open(os.path.join(directory, str(rank) + ".json"), "w") as output:
            json.dump({"mean": mean.tolist(), "gradient": value.grad.tolist()}, output)
    finally:
        dist.destroy_process_group()


class TestLFQDistributedMean(unittest.TestCase):
    def check_ranks(self, warm_before_init):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = "file://" + os.path.join(directory, "rendezvous")
            mp.spawn(run_rank, args=(rendezvous, directory, warm_before_init), nprocs=2, join=True)
            for rank in range(2):
                with open(os.path.join(directory, str(rank) + ".json")) as result:
                    data = json.load(result)
                torch.testing.assert_close(torch.tensor(data["mean"]), torch.tensor([2., 3.]))
                torch.testing.assert_close(torch.tensor(data["gradient"]), torch.ones(2))

    def test_global_mean_and_collective_backward(self):
        self.check_ranks(False)

    def test_process_group_initialized_after_first_helper_call(self):
        self.check_ranks(True)


if __name__ == "__main__":
    unittest.main()
