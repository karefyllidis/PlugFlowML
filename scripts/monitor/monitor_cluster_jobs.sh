#!/bin/bash
# Live snapshot of HydrAI SLURM data-generation cluster jobs (Main_2 chunks).
# Run from repo root at any time during or after a job:
#   bash scripts/monitor/monitor_cluster_jobs.sh
#
# Optional: watch mode (refresh every 30 s):
#   watch -n 30 "bash scripts/monitor/monitor_cluster_jobs.sh"

set -euo pipefail

SEP="============================================================"

# ── Root and config ──────────────────────────────────────────
echo ""
echo "$SEP"
echo "  HydrAI cluster job monitor  ($(date))"
echo "$SEP"

if [[ -f logs/RUN_ROOT.txt ]]; then
  echo ""
  echo "── RUN_ROOT ─────────────────────────────────────────────"
  cat logs/RUN_ROOT.txt
else
  echo "[WARN] logs/RUN_ROOT.txt not found"
fi

# ── SLURM queue ──────────────────────────────────────────────
echo ""
echo "── SLURM queue ──────────────────────────────────────────"
if command -v squeue &>/dev/null; then
  squeue -u "$(whoami)" 2>/dev/null || echo "  (squeue unavailable)"
else
  echo "  (squeue not found)"
fi

# ── Per-task progress (aggregate across all JSON files) ──────
echo ""
echo "── Per-task progress ────────────────────────────────────"

total_completed=0
total_successful=0
total_failed=0
total_tasks=0
tasks_done=0
tasks_running=0
tasks_empty=0

json_files=( logs/data_generation_progress_task_*.json )
if [[ -e "${json_files[0]}" ]]; then
  for f in "${json_files[@]}"; do
    task_id=$(basename "$f" | sed 's/data_generation_progress_task_//' | sed 's/\.json//')
    # parse JSON fields with grep+sed (no jq required)
    completed=$(grep '"completed"'   "$f" 2>/dev/null | sed 's/[^0-9]//g' | head -1)
    total=$(grep '"total_this_task"' "$f" 2>/dev/null | sed 's/[^0-9]//g' | head -1)
    successful=$(grep '"successful"' "$f" 2>/dev/null | sed 's/[^0-9]//g' | head -1)
    failed=$(grep '"failed"'        "$f" 2>/dev/null | sed 's/[^0-9]//g' | head -1)
    pct=$(grep '"percent_this_task"' "$f" 2>/dev/null | grep -o '[0-9.]*' | head -1)

    completed=${completed:-0}
    total=${total:-0}
    successful=${successful:-0}
    failed=${failed:-0}
    pct=${pct:-0}

    total_tasks=$((total_tasks + 1))
    total_completed=$((total_completed + completed))
    total_successful=$((total_successful + successful))
    total_failed=$((total_failed + failed))

    if [[ "$total" -eq 0 ]]; then
      tasks_empty=$((tasks_empty + 1))
    elif [[ "$completed" -ge "$total" ]]; then
      tasks_done=$((tasks_done + 1))
    else
      tasks_running=$((tasks_running + 1))
      printf "  task %-3s  %3s/%-3s  (%5s%%)  ok=%-3s fail=%s\n" \
        "$task_id" "$completed" "$total" "$pct" "$successful" "$failed"
    fi
  done

  echo ""
  echo "── Totals ───────────────────────────────────────────────"
  echo "  Tasks found    : $total_tasks"
  echo "  Tasks with work: $((tasks_done + tasks_running))  ($tasks_empty idle/empty)"
  echo "  Tasks finished : $tasks_done"
  echo "  Tasks running  : $tasks_running"
  echo "  Simulations    : completed=$total_completed  ok=$total_successful  fail=$total_failed"
else
  echo "  No progress JSON files found yet."
fi

# ── Output files count ───────────────────────────────────────
echo ""
echo "── Output files ─────────────────────────────────────────"
n_pkl=$(find data/training -maxdepth 2 -name 'training_data_complete_*.pkl' 2>/dev/null | wc -l)
n_partial=$(find data/training -maxdepth 2 -name 'training_data_partial_*.pkl' 2>/dev/null | wc -l)
n_meta=$(find data/training -maxdepth 2 -name 'metadata_*.json' 2>/dev/null | wc -l)
echo "  Complete pkl  : $n_pkl"
echo "  Partial pkl   : $n_partial"
echo "  Metadata json : $n_meta"

# ── srun errors ──────────────────────────────────────────────
if [[ -f logs/srun_step.err ]] && [[ -s logs/srun_step.err ]]; then
  echo ""
  echo "── srun_step.err (non-empty) ────────────────────────────"
  tail -10 logs/srun_step.err
fi

echo ""
echo "$SEP"
echo "  Tip: tail -f logs/main2_task_0.log"
echo "$SEP"
echo ""
