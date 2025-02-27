#!/bin/bash
# 项目根目录的脚本入口

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示帮助信息
show_help() {
    echo "技能路径匹配项目脚本"
    echo "用法: $0 [命令] [参数]"
    echo ""
    echo "可用命令:"
    echo "  train       训练模型"
    echo "  evaluate    评估模型"
    echo "  help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 train      # 训练模型"
    echo "  $0 evaluate   # 评估模型"
}

# 主函数
main() {
    case "$1" in
        train)
            bash "$PROJECT_ROOT/src/training/scripts/train_model.sh"
            ;;
        evaluate)
            bash "$PROJECT_ROOT/src/evaluation/scripts/evaluate.sh"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "错误: 未知命令 '$1'"
            show_help
            exit 1
            ;;
    esac
}

# 如果没有参数，显示帮助信息
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# 执行主函数
main "$@"
