#!/usr/bin/env bash
# sc-25690 (epic 25678 terminal evidence): the Iris-3B SceneWorks worker real-weight smokes on the
# SECOND Mac (nax-macos-2) through the production job path — generation, depth and restoration
# jobs, and the generation (LoRA / LoKr cancel→resume / two-rank DDP), depth and restoration
# training jobs with their trained artifacts used by a follow-up job. Dispatch-only, gates nothing
# (FEATURE_DEVELOPMENT.md "Gate teardown").
#
# Safety, in order, before any GPU work:
#   * the runner must be nax-macos-2 — the dev Mac (nax-macos) is never used, whatever label routed
#     the job here;
#   * >= 128 GB unified memory, and `memory_pressure` free GB >= the test's cap + 10 GB headroom
#     right before EACH test;
#   * every test binary runs under an external phys_footprint guard summed over its whole process
#     tree (the two-rank DDP smoke spawns rank processes) that SIGTERMs the tree at a stated cap and
#     at a wall-clock limit — the only kill path; MLX is never SIGKILLed mid-buffer from here.
#
# The depth and restoration jobs (inference AND training) run against an HF hub that holds ONLY
# the speridlabs/iris-3b snapshot (+ restoration's three training resources): the Qwen3-VL text
# encoder is provably absent from what they can resolve.
#
# Inputs (env): IRIS_SMOKE_FILTER = all | inference | training | <one test name> | cleanup.
set -euo pipefail

out="${IRIS_EVIDENCE_DIR:?}"
mkdir -p "$out"
filter="${IRIS_SMOKE_FILTER:-all}"
sha="$(git rev-parse HEAD)"
target="$HOME/sceneworks-iris-campaign-target/$sha"

# ---- 1. the box -----------------------------------------------------------------------------------
mem_bytes="$(sysctl -n hw.memsize)"
mem_gb=$((mem_bytes / 1000000000))
{
  echo "runner=${RUNNER_NAME:-unknown} source_sha=$sha filter=$filter"
  sysctl -n hw.model machdep.cpu.brand_string
  echo "unified memory: $mem_gb GB"
  /usr/bin/memory_pressure -Q || true
  df -h "$HOME" | tail -1
} | tee "$out/host.txt"
if [[ "${RUNNER_NAME:-}" != "nax-macos-2" ]]; then
  echo "::error::refusing: this lane runs only on nax-macos-2 (landed on ${RUNNER_NAME:-unknown}); no GPU work was started"
  exit 1
fi
if (( mem_gb < 128 )); then
  echo "::error::refusing: $mem_gb GB unified memory (< 128 GB)"
  exit 1
fi

if [[ "$filter" == "cleanup" ]]; then
  # Only this lane's own build trees.
  rm -rf "$HOME/sceneworks-iris-campaign-target"
  echo "removed $HOME/sceneworks-iris-campaign-target" | tee -a "$out/host.txt"
  exit 0
fi

# ---- 2. the weights (offline at test time) --------------------------------------------------------
# The box's task-owned real-weight hub (inference's Iris lanes already materialized the generation
# backbone, upscaler/, Qwen3-VL and the restoration training resources there). The only Iris task
# export not yet resident is depth/; it is fetched here, pinned, through the HF cache layout.
hub="$HOME/sceneworks-rw-weights/hub"
iris_rev=7445443349bc9abe3c96f01ff793e2098ca012b3
iris_repo_dir="$hub/models--speridlabs--iris-3b"
qwen_dir="$hub/models--Qwen--Qwen3-VL-4B-Instruct"
critic_dir="$hub/models--laion--CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
vgg_dir="$hub/models--timm--vgg16.tv_in1k"
lpips_dir="$hub/models--vivym--lpips"
python3.12 --version | grep -E '^Python 3\.12\.'
python3.12 -m venv "$RUNNER_TEMP/iris-hf-venv"
"$RUNNER_TEMP/iris-hf-venv/bin/python" -m pip install --disable-pip-version-check --no-input "huggingface_hub==0.36.0" >/dev/null
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
IRIS_HUB="$hub" IRIS_REV="$iris_rev" "$RUNNER_TEMP/iris-hf-venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev, patterns in [
    ("speridlabs/iris-3b", os.environ["IRIS_REV"], ["config.yaml", "model.safetensors", "depth/*", "upscaler/*"]),
    ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17",
     ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
      "model.safetensors.index.json", "model-*.safetensors"]),
    ("laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup", "9f3e8ee3f383c672388d9178afe70af9e63ac9df", None),
    ("timm/vgg16.tv_in1k", "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f", None),
    ("vivym/lpips", "270571f1fb2a2c4c5f920cec96cd87838c345982", None),
]:
    snap = os.path.join(os.environ["IRIS_HUB"], "models--" + repo.replace("/", "--"), "snapshots", rev)
    # Resident = every literal pattern present and every directory pattern non-empty; a glob-only
    # repo (no patterns) = a non-empty snapshot. A partial snapshot is completed by the fetch.
    def present(p):
        path = os.path.join(snap, p.split("*")[0].rstrip("/"))
        return os.path.isdir(path) and bool(os.listdir(path)) if p.endswith("/*") else (
            os.path.exists(path) if "*" not in p else os.path.isdir(snap))
    have = all(present(p) for p in patterns) if patterns else (
        os.path.isdir(snap) and bool(os.listdir(snap)))
    if have:
        print(f"resident: {repo}@{rev}")
        continue
    print(f"fetching: {repo}@{rev} {patterns or 'whole repo'}", flush=True)
    snapshot_download(repo_id=repo, revision=rev, cache_dir=os.environ["IRIS_HUB"],
                      allow_patterns=patterns, token=False, max_workers=4)
PY
snap="$iris_repo_dir/snapshots/$iris_rev"
for f in config.yaml model.safetensors depth/config.yaml depth/model.safetensors depth/empty_prompt.safetensors \
         upscaler/config.yaml upscaler/model.safetensors upscaler/empty_prompt.safetensors; do
  test -s "$snap/$f" || { echo "::error::missing $snap/$f"; exit 1; }
done
test -d "$qwen_dir/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"

# The encoder-free hub for the depth / restoration jobs: the Iris repo (+ restoration's training
# resources), and nothing else.
no_encoder="$RUNNER_TEMP/iris-hub-no-encoder"
rm -rf "$no_encoder"
mkdir -p "$no_encoder"
for d in "$iris_repo_dir" "$critic_dir" "$vgg_dir" "$lpips_dir"; do
  ln -s "$d" "$no_encoder/$(basename "$d")"
done
if find -L "$no_encoder" -maxdepth 1 -iname '*qwen*' | grep -q .; then
  echo "::error::the encoder-free hub resolves a Qwen repository"; exit 1
fi
find "$no_encoder" -maxdepth 1 -exec ls -ld {} + | tee "$out/no-encoder-hub.txt"

# ---- 3. build (run-scoped per source SHA; unstripped, rust-lang/rust#157750) ---------------------
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
build_start=$(date +%s)
cargo test --locked --release -p sceneworks-worker --lib --no-run
bin="$(cargo test --locked --release -p sceneworks-worker --lib --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "sceneworks_worker":
        print(m["executable"])' | tail -1)"
test -x "$bin"
echo "build ${SECONDS}s total, $(( $(date +%s) - build_start ))s cargo; binary $bin" | tee "$out/build.txt"

# ---- 4. guarded runs ------------------------------------------------------------------------------
tree_pids() { # $1 root pid -> the root and every descendant
  local p=$1
  echo "$p"
  for c in $(pgrep -P "$p" 2>/dev/null || true); do tree_pids "$c"; done
}
failed=0
run_guarded() { # $1 test path, $2 cap GB, $3 wall s, $4 hub
  local test=$1 cap_gb=$2 limit_s=$3 hub_dir=$4 name="${1##*::}"
  local log="$out/$name.log" free_pct free_gb
  free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
  free_gb=$(( mem_gb * ${free_pct:-0} / 100 ))
  if [[ -z "$free_pct" || "$free_gb" -lt $((cap_gb + 10)) ]]; then
    echo "::error::$name not started: ${free_gb} GB free (memory_pressure ${free_pct:-?}%) < cap ${cap_gb} + 10 GB" | tee "$log"
    failed=1
    return
  fi
  local cap_kb=$((cap_gb * 1000 * 1000)) start pid peak=0 fp total stop=""
  start=$(date +%s)
  HF_HUB_CACHE="$hub_dir" HF_HUB_OFFLINE=1 TMPDIR="$RUNNER_TEMP" \
    "$bin" "$test" --ignored --exact --nocapture --test-threads=1 > "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    total=0
    for p in $(tree_pids "$pid"); do
      fp=$(footprint -p "$p" 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3; if (u=="GB") v*=1048576; else if (u=="MB") v*=1024; print int(v); exit}' || true)
      total=$((total + ${fp:-0}))
    done
    (( total > peak )) && peak=$total
    if [[ -z "$stop" && "$total" -gt "$cap_kb" ]]; then
      stop="footprint ${total} KB > ${cap_gb} GB cap"
      echo "GUARD: $stop, SIGTERM" | tee -a "$out/guard.txt"
      # shellcheck disable=SC2046 # one pid per word
      kill -TERM $(tree_pids "$pid") 2>/dev/null || true
    fi
    if [[ -z "$stop" && $(( $(date +%s) - start )) -gt "$limit_s" ]]; then
      stop="wall time over ${limit_s}s"
      echo "GUARD: $stop, SIGTERM" | tee -a "$out/guard.txt"
      # shellcheck disable=SC2046 # one pid per word
      kill -TERM $(tree_pids "$pid") 2>/dev/null || true
    fi
    sleep 1
  done
  set +e
  wait "$pid"
  local rc=$?
  set -e
  cat "$log"
  echo "[[IRIS_MLX_SMOKE]] $name exit=$rc peak_tree_phys_footprint_gb=$(awk -v k="$peak" 'BEGIN{printf "%.2f", k/1048576}') cap_gb=$cap_gb wall_s=$(( $(date +%s) - start )) hub=$hub_dir ${stop:+stopped=\"$stop\"}" \
    | tee -a "$out/summary.txt"
  if [[ "$rc" != 0 ]] || ! grep -q "test result: ok. 1 passed" "$log"; then
    echo "::error::$test did not run exactly one passing test"
    failed=1
  fi
}

inference_set=(
  "iris_gpu_smoke::iris_worker_gpu_smoke_native_prompt_batch 45 900 $hub"
  "depth_jobs::gpu_smoke::iris_worker_gpu_smoke_depth 30 600 $no_encoder"
  "iris_gpu_smoke::iris_worker_gpu_smoke_restore 40 900 $no_encoder"
)
training_set=(
  "iris_training_gpu_smoke::iris_training_gpu_smoke 45 1200 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_lokr_cancel_resume 45 1200 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_two_rank_ddp 80 1500 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_depth 90 1200 $no_encoder"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_restore 50 1500 $no_encoder"
)
case "$filter" in
  all) selected=("${inference_set[@]}" "${training_set[@]}") ;;
  inference) selected=("${inference_set[@]}") ;;
  training) selected=("${training_set[@]}") ;;
  *)
    selected=()
    for row in "${inference_set[@]}" "${training_set[@]}"; do
      [[ "${row%% *}" == *"::$filter" ]] && selected+=("$row")
    done
    ;;
esac
if (( ${#selected[@]} == 0 )); then
  echo "::error::filter '$filter' selects no Iris smoke"
  exit 1
fi
for row in "${selected[@]}"; do
  # shellcheck disable=SC2086
  run_guarded $row
done
exit "$failed"
