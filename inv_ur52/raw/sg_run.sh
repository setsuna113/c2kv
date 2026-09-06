SG=/home/zhuyuhan/project/kvoffload-sglang
echo "@@@@@ committed qwen3 raw_kv markers"
git -C "$SG" show 7de9e81051a31aa02e635c8af86f6ca100cc7f0c:python/sglang/srt/models/qwen3.py | grep -n "raw_kv_position_mode\|pre_rope" | head -5
echo "@@@@@ committed scheduler markers"
git -C "$SG" show 7de9e81051a31aa02e635c8af86f6ca100cc7f0c:python/sglang/srt/managers/scheduler.py | grep -n "append_masked_w2\|raw_kv_position_mode\|repair_advances_logical_position" | head -10
echo "@@@@@ committed io_struct markers"
git -C "$SG" show 7de9e81051a31aa02e635c8af86f6ca100cc7f0c:python/sglang/srt/managers/io_struct.py | grep -n "already_rotated\|raw_kv_position_mode\|gist_projection\|repair_mode" | head -10
echo "@@@@@ committed protocol markers"
git -C "$SG" show 7de9e81051a31aa02e635c8af86f6ca100cc7f0c:python/sglang/srt/entrypoints/openai/protocol.py | grep -n "already_rotated\|gist_projection\|repair" | head -10
echo "@@@@@@@@@ ANALYZE"
python3 /tmp/zh/analyze1.py
