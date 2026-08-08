# ml-metrics — the training curves of every ml-train run

One file per run: `run-<run_number>.jsonl`, the metrics.jsonl the job
produced (loss points every few hundred steps; probe points at each
eval interval). Read by status.html so a completed run keeps its charts.

Runs before #47 were backfilled from locally harvested artifacts; from
#47 on, scripts/archive_run_metrics.sh writes this branch automatically
when a job ends. This branch is bookkeeping and is never merged to main.
