import torch.distributed as dist


def train_epoch(start_epoch, num_epochs, model, dataloader, optimizer, rank, world_size):
    for epoch in range(start_epoch, num_epochs):
        # Ensure all processes start the epoch at the same time
        dist.barrier()
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        # Collective operation: Summing up all the losses to calculate average
        loss_tensor = torch.tensor([loss.item()]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:  # Only the master process logs the information
            print(f"Avg loss across all processes: {loss_tensor.item() / world_size}")

        # Synchronize again before starting the next epoch or doing any checkpointing
        dist.barrier()
