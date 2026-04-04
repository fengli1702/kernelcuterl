
# Choose model format: "qwen3.5" or "qwen3next"
MODEL_FORMAT=${MODEL_FORMAT:-"qwen3.5"}
# Enable MTP speculative decoding: "true" or "false"
USE_MTP=${USE_MTP:-"true"}

TOKENIZER_ARGS=""
TOKENIZER=${TOKENIZER:-"/home/data/public/data/EvaluationData_LFS/tokenizer/qwen35-posttrain-think"}
if [ "$MODEL_FORMAT" = "qwen3.5" ]; then
    # Qwen3.5 format: use opensource sglang
    export PYTHONPATH=/cpfs01/user/huiqiang.zzh/codespace/0215_opensource/sglang/python:$PYTHONPATH
    FP8_RAW=/cpfs01/data/shared/Group-m6/wangzhihai.wzh/ckpts/mcore-qwen3.5/turbo.v1/SFT/checkpoint/SFT_qwen3.5-35b-s2opd-final-mtp-rollout5-qwen3.5vl_35b_256kcpt_s2opd-final-mtp-rollout5-bf16-turbo.v1-gqa2-mp1-pp2-lr7e-6-minlr7e-7-vlrd0.9-1.0-audiord--iters10354-decay10354-warmup100-bs2048-gpu1024-seqlen262144-fp8-000/huggingface/iter_0003584_opensource
else
    # Qwen3Next format: use sglang-internal
    export PYTHONPATH=/cpfs01/user/huiqiang.zzh/codespace/0306_mtp_training/sglang-internal/python:$PYTHONPATH
    FP8_RAW=/cpfs01/user/wangzhihai.wzh/251129_RFT_Models/qwen3.5_35b_s2_opd_cand2/hf_model
    TOKENIZER_ARGS="--tokenizer-path ${TOKENIZER}"
fi

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth
export ACCL_LOAD_BALANCE=1
export ACCL_LOW_LATENCY_OPTIMIZE=1
export NCCL_CUMEM_ENABLE=1
export GLOO_SOCKET_IFNAME=eth0
export USE_CHAIN_SPECULATIVE_SAMPLING=1



MTP_ARGS=""
if [ "$USE_MTP" = "true" ]; then
    MTP_ARGS="--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
fi

python -m sglang.launch_server \
    --model $FP8_RAW \
    --dtype bfloat16 \
    --tp-size 2 \
    --dp-size 1 \
    --disable-radix-cache \
    --mem-fraction-static 0.75 \
    --max-running-requests 128 \
    --attention-backend fa3 \
    --enable-memory-saver \
    --chunked-prefill-size 8192 \
    --num-continuous-decode-steps 9 \
    --disable-outlines-disk-cache \
    --cuda-graph-max-bs 128 \
    --mamba-full-memory-ratio 1 \
    ${MTP_ARGS} \
    ${TOKENIZER_ARGS}
