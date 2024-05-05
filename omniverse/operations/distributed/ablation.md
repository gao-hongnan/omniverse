# Ablations

No money, so can only do a few ablations to learn about distributed systems.

## No Distributed Barrier

If you run:

```bash
python omnixamples/distributed/a_raw/ablations.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://" \
    --run_with_no_barrier
```

Which is invoking a simple function:

```python
def run_with_no_barrier(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    results = []

    logger.info("I HAVE NO BARRIER DUDE!")

    results.append([1, 2, 3])

    logger.info(f"Results: {results}")
```

At times you may encounter the below:

```python
                    INFO     2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                                                                                                                                                      ablations.py:32
                    INFO     2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                                                                                                                                                      ablations.py:32
                    INFO     2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                                                                                                                                                         ablations.py:36
                    INFO     2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                                                                                                                                                      ablations.py:32
                    INFO     2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                                                                                                                                                         ablations.py:36
2024-05-05 13:29:55 INFO     2024-05-05 13:29:55 [INFO]: {                                                                                                                                                                                            ablations.py:28
                                 "master_addr": "localhost",
                                 "master_port": "29500",
                                 "nnodes": 1,
                                 "nproc_per_node": 4,
                                 "node_rank": 0,
                                 "world_size": 4,
                                 "backend": "gloo",
                                 "init_method": "env://",
                                 "global_rank": 3,
                                 "local_world_size": 4,
                                 "local_rank": 3,
                                 "hostname": "Hongnans-Mac-mini.local",
                                 "process_id": 20647
                             }
                    INFO     2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                                                                                                                                                         ablations.py:36
                    INFO     2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                                                                                                                                                      ablations.py:32
                    INFO     2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]
```

You see that even when printing the `results` is after the
`I HAVE NO BARRIER DUDE!` message, the results are printed before the message.
This is because there is no barrier to synchronize the processes. So you can add
a `torch.distributed.barrier()` to synchronize the processes before printing the
results.
