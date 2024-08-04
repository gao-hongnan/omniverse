# A Simple Distributed Walkthrough

Assuming 2 nodes. One master and one worker - both compute nodes. Go into a
cluster say SLURM head node and ssh into both compute nodes to run the following
commands on each node.

```bash
bash omnixamples/distributed/01_raw/scripts/01_demo_start_master.sh
```

```bash
bash omnixamples/distributed/01_raw/scripts/01_demo_start_worker.sh
```

For more information, please refer to the
[distributed](https://www.gaohongnan.com/operations/distributed/02_basics.html)
page.
