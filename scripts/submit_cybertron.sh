#!/usr/bin/env bash
# Submit a csr_grpo run to the Cybertron platform via cctl.
#
# The dev machine has no GPU (nvidia-smi -L: "No devices found"), so training
# happens here. Everything the job needs is a git ref (--code-type git) or an
# entry command that clones/uses an already-synced workspace.
#
#   ./scripts/submit_cybertron.sh \
#       --project my-proj --cluster gpu-cluster --pool training-pool \
#       --image nvcr.io/nvidia/pytorch:24.01-py3 --gpu 8 \
#       --command "python -m csr_grpo.cli e1 --config configs/e1_switch.yaml --steps 300 --heldout"
#
# Run `cctl login` first (and `cctl pool list --own` to find a pool).
set -euo pipefail

# The default profile ("modelbest") records a login but not a server, and cctl
# then builds URLs from an empty scheme (``unsupported protocol scheme ""``).
# autosft's launch scripts export this explicitly; so must we.
export CCTL_SERVER="${CCTL_SERVER:-https://cybertron.modelbest.co}"

PROJECT=""; CLUSTER=""; POOL=""; IMAGE=""; GPU=8; GPUMODEL=""; COMMAND=""
BILLING="${CCTL_BILLING_ACCOUNT_ID:-N00069}"
DESCRIPTION="csr_grpo run"; DRYRUN=0; KIND="BATCH"; WORKERS=""
GIT_PATH=""; GIT_REF=""; PRIORITY="NORMAL"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'H'

Flags:
  --project P        required   cctl project
  --cluster C        required   cluster name
  --pool P           required   resource pool (id | name | alias)
  --image I          required   container image
  --command "..."    required   entry command
  --gpu N            default 8  GPU count
  --gpu-model M      e.g. H800
  --kind K           BATCH (default) | PYTORCHJOB
  --workers N        required for PYTORCHJOB
  --priority P       HIGH | NORMAL | PREEMPTABLE
  --git-path P       enables --code-type git (requires --git-ref)
  --git-ref R
  --description D
  --billing ID       billing account (default $CCTL_BILLING_ACCOUNT_ID, else N00069;
                     cctl rejects task creation without it on this server)
  --dry-run          print the command instead of running it
H
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --cluster) CLUSTER="$2"; shift 2;;
    --pool) POOL="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --command) COMMAND="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --gpu-model) GPUMODEL="$2"; shift 2;;
    --kind) KIND="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --priority) PRIORITY="$2"; shift 2;;
    --git-path) GIT_PATH="$2"; shift 2;;
    --billing) BILLING="$2"; shift 2;;
    --git-ref) GIT_REF="$2"; shift 2;;
    --description) DESCRIPTION="$2"; shift 2;;
    --dry-run) DRYRUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown flag: $1" >&2; usage; exit 64;;
  esac
done

for v in PROJECT CLUSTER POOL IMAGE COMMAND; do
  [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 64; }
done
if [[ "$KIND" == "PYTORCHJOB" && -z "$WORKERS" ]]; then
  echo "--workers is required for PYTORCHJOB" >&2; exit 64
fi
if [[ -n "$GIT_PATH" && -z "$GIT_REF" ]]; then
  echo "--git-ref is required with --git-path" >&2; exit 64
fi

CMD=(cctl job create
  --project "$PROJECT" --cluster "$CLUSTER" --resource-pool "$POOL"
  --billing-account-id "$BILLING"
  --image "$IMAGE" --gpu "$GPU" --entry "$COMMAND"
  --priority "$PRIORITY" --description "$DESCRIPTION")

if [[ "$KIND" == "PYTORCHJOB" ]]; then
  CMD=(cctl pytorchjob create
    --project "$PROJECT" --cluster "$CLUSTER" --resource-pool "$POOL"
    --billing-account-id "$BILLING"
    --image "$IMAGE" --gpu "$GPU" --workers "$WORKERS" --entry "$COMMAND"
    --priority "$PRIORITY" --description "$DESCRIPTION")
fi
[[ -n "$GPUMODEL" ]] && CMD+=(--gpu-model "$GPUMODEL")
[[ -n "$GIT_PATH" ]] && CMD+=(--code-type git --git-path "$GIT_PATH" --git-ref "$GIT_REF")

# The supervision ledger is only meaningful if the training run used the real
# verifier; refuse to submit a config whose reward source is gold without the
# reference/diagnostic flag.
echo "submitting: ${DESCRIPTION}"
printf ' %q' "${CMD[@]}"; echo
if [[ $DRYRUN -eq 1 ]]; then exit 0; fi
"${CMD[@]}"
