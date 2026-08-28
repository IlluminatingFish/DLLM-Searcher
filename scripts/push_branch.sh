#!/bin/bash
# 创建新 branch 并将当前代码 push 到 GitHub
# 用法:
#   bash push_branch.sh <branch-name> [commit-message]
# 示例:
#   bash push_branch.sh dev-aug27
#   bash push_branch.sh dev-aug27 "add llama3 sft eval scripts"

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="$1"
MSG="${2:-update: $(date '+%Y-%m-%d %H:%M')}"

if [ -z "$BRANCH" ]; then
    echo "用法: bash push_branch.sh <branch-name> [commit-message]"
    echo "示例: bash push_branch.sh dev-aug27 \"add llama3 eval\""
    exit 1
fi

echo "========================================"
echo "仓库: $(git remote get-url origin)"
echo "Branch: $BRANCH"
echo "Commit message: $MSG"
echo "========================================"

# 查看当前未提交的改动
echo ""
echo "=== 当前改动 ==="
git status --short

# 创建并切换到新 branch（如果已存在则要求确认）
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo ""
    echo "⚠️  Branch '$BRANCH' 已存在！这会把代码推到这个旧 branch 上。"
    read -p "确认继续？(y/N) " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "已取消。请换一个新的 branch 名字再试。"
        exit 1
    fi
    git checkout "$BRANCH"
else
    echo ""
    echo "创建新 branch: $BRANCH"
    git checkout -b "$BRANCH"
fi

# 添加所有改动（遵循 .gitignore）
git add -A

# 如果没有新改动就跳过 commit
if git diff --cached --quiet; then
    echo "没有新改动需要提交，直接 push..."
else
    git commit -m "$MSG"
fi

# Push 到 GitHub
echo ""
echo "Push 到 origin/$BRANCH ..."
git push -u origin "$BRANCH"

echo ""
echo "✅ 完成！Branch '$BRANCH' 已上传到 GitHub"
echo "   https://github.com/IlluminatingFish/DLLM-Searcher/tree/$BRANCH"
