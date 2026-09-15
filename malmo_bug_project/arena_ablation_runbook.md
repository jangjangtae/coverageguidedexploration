# Arena ablation runbook

## Conditions

- `RND_CAE_FINAL`: accepted Full implementation, unchanged.
- `RND_CAE_SPATIAL`: direct `(int(XPos), int(ZPos))` cell key only; matched
  CAE key-weight budget `0.405`; stagnation enabled.
- `RND_CAE_NO_STAG`: all accepted Full keys and weights; stagnation disabled.

Use seeds `1..10` and `100000` steps to pair Spatial and NoStag with the Full
condition. The 200-step folders are integration smoke tests only. Run the
Python commands below from `MalmoPlatform/malmo_bug_project`.

## Recommended five-client execution

Start five Minecraft/Malmo clients from the Malmo `Minecraft` directory:

```bash
./launchClient.sh -port 10000
./launchClient.sh -port 10001
./launchClient.sh -port 10002
./launchClient.sh -port 10003
./launchClient.sh -port 10004
```

Assign two paired seeds to each port. Each worker executes Spatial and NoStag
sequentially, for four 100k runs per port. Use a distinct worker subdirectory
because concurrent trainer processes otherwise overwrite the root experiment
manifest. Ports `10001` and `10003` use the reverse condition order to avoid
confounding every Spatial run with the first half of the wall-clock schedule.

```bash
python3 -m agents.train_arena_issue10_final \
  --algo ablations \
  --port 10000 \
  --seeds 1,2 \
  --steps 100000 \
  --log-root ./logs_arena_ablation/worker_p10000 \
  --split-log-by-algo \
  --cooldown 100

python3 -m agents.train_arena_issue10_final \
  --algo ablations_reverse \
  --port 10001 \
  --seeds 3,4 \
  --steps 100000 \
  --log-root ./logs_arena_ablation/worker_p10001 \
  --split-log-by-algo \
  --cooldown 100

python3 -m agents.train_arena_issue10_final \
  --algo ablations \
  --port 10002 \
  --seeds 5,6 \
  --steps 100000 \
  --log-root ./logs_arena_ablation/worker_p10002 \
  --split-log-by-algo \
  --cooldown 100

python3 -m agents.train_arena_issue10_final \
  --algo ablations_reverse \
  --port 10003 \
  --seeds 7,8 \
  --steps 100000 \
  --log-root ./logs_arena_ablation/worker_p10003 \
  --split-log-by-algo \
  --cooldown 100

python3 -m agents.train_arena_issue10_final \
  --algo ablations \
  --port 10004 \
  --seeds 9,10 \
  --steps 100000 \
  --log-root ./logs_arena_ablation/worker_p10004 \
  --split-log-by-algo \
  --cooldown 100
```

One 100k Arena run takes about 9.5 hours without contention. Five clients need
about 38 hours ideally for four runs per worker; allow roughly 40--55 hours for
CPU and I/O contention.

Do not count a mission-start or native-wrapper failure as a zero-bug run.
Rerun that condition with the same seed and retain only the completed summary.

## Analysis

After all runs complete:

```bash
python3 analyze_arena_ablation.py \
  --roots logs_arena_issue10_final logs_arena_ablation \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --out analysis_arena_ablation
```

The analysis writes run-level metrics, aggregate summaries, paired differences
against Full, and per-fault discovery rates. Pairs with unequal training
horizons are excluded automatically.
