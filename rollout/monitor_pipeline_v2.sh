#!/bin/bash
# Pipeline V2 监控脚本

RESULT_DIR="/cpfs01/user/lidaifeng.ldf/KernelRL/rollout/results"
LOG_FILE="$RESULT_DIR/rollout_pipeline_v2.log"
CHECKPOINT_FILE="$RESULT_DIR/kernelbench_rollout_pipeline_v2.jsonl.checkpoint.jsonl"
FINAL_FILE="$RESULT_DIR/kernelbench_rollout_pipeline_v2.jsonl"
TOTAL=270

echo "========================================"
echo "   Pipeline Rollout V2 Monitor"
echo "   (改进的代码提取逻辑)"
echo "========================================"
echo ""

# 检查进程状态
if ps aux | grep -v grep | grep "single_turn_rollout.py.*pipeline_v2" > /dev/null; then
    echo "✅ Rollout进程: 运行中"
else
    echo "❌ Rollout进程: 未运行"
fi

if ps aux | grep -v grep | grep "sglang.launch_server" > /dev/null; then
    echo "✅ SGLang服务器: 运行中"
else
    echo "❌ SGLang服务器: 未运行"
fi

echo ""
echo "----------------------------------------"
echo "进度信息:"
echo "----------------------------------------"

# 检查checkpoint文件
if [ -f "$CHECKPOINT_FILE" ]; then
    COMPLETED=$(wc -l < "$CHECKPOINT_FILE")
    PERCENT=$(echo "scale=2; $COMPLETED * 100 / $TOTAL" | bc)
    echo "✓ 已完成: $COMPLETED / $TOTAL ($PERCENT%)"

    FILE_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    echo "✓ Checkpoint大小: $FILE_SIZE"
elif [ -f "$FINAL_FILE" ]; then
    COMPLETED=$(wc -l < "$FINAL_FILE")
    PERCENT=$(echo "scale=2; $COMPLETED * 100 / $TOTAL" | bc)
    echo "✓ 已完成: $COMPLETED / $TOTAL ($PERCENT%)"
    echo "✓ 状态: 完成"
else
    echo "⏳ 等待第一个checkpoint..."
fi

echo ""
echo "----------------------------------------"
echo "最新日志 (最后10行):"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE" | grep -E "(Rollout:|INFO|ERROR|WARNING)" || tail -3 "$LOG_FILE"
else
    echo "日志文件尚未创建"
fi

echo ""
echo "----------------------------------------"
echo "改进内容:"
echo "----------------------------------------"
echo "✅ 提取最后一个包含ModelNew的代码块"
echo "✅ 自动获取</think>后的最终修正版本"
echo "✅ 平均提取长度预计提升65%"

echo ""
echo "----------------------------------------"
echo "快捷命令:"
echo "----------------------------------------"
echo "实时日志: tail -f $LOG_FILE"
echo "查看进度: watch -n 5 bash $0"
echo "停止任务: pkill -f 'pipeline_v2'"
echo "========================================"
