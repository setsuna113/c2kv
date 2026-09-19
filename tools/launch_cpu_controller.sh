#!/usr/bin/env bash
set -e

if (( $# == 0 )); then
    echo "usage: launch_cpu_controller.sh <python> <driver> [args...]" >&2
    exit 64
fi

for setup in "${C2KV_CANN_SETUP:-/usr/local/Ascend/cann-8.5.0/set_env.sh}" \
             "${C2KV_NNAL_SETUP:-/usr/local/Ascend/nnal/atb/set_env.sh}"; do
    if [[ ! -r "$setup" ]]; then
        echo "missing Ascend environment: $setup" >&2
        exit 1
    fi
    source "$setup"
done

# The controller runs CPU inference, but Transformers may import torch_npu to
# inspect availability. Keep backend autoload disabled and make its libraries
# discoverable without changing the embedding device.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec "$@"
