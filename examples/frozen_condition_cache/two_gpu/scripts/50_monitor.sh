#!/bin/bash
# Sample host/GPU/shm/cache-dir usage every ${INTERVAL:-5}s until killed.
: "${MONITOR_GPU_IDS:?Set the two authorized physical GPU IDs}"
OUT="$1"; CACHE_ROOT="$2"; INTERVAL="${INTERVAL:-5}"
echo "wall,mem_used_bytes,mem_available_bytes,shm_used_bytes,cache_dir_bytes,cache_run_dirs,gpu_mem_used_mib_total,gpu_mem_used_mib_max" > "$OUT"
while true; do
    now=$(date +%s.%N)
    read -r used avail < <(free -b | awk '/^Mem:/ {print $3, $7}')
    shm=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
    cache=$(du -sb "$CACHE_ROOT" 2>/dev/null | cut -f1); cache=${cache:-0}
    runs=$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    gpu=$(nvidia-smi -i "$MONITOR_GPU_IDS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1; if($1>m)m=$1} END {if(NR>0) print (s+0) "," (m+0); else print ","}')
    echo "$now,$used,$avail,${shm:-0},$cache,$runs,${gpu:-0,0}" >> "$OUT"
    sleep "$INTERVAL"
done
