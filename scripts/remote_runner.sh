#!/usr/bin/env bash
# GPU 机器上的任务执行器：轮询本分支的 jobs/queue/，按文件名顺序执行未完成任务，
# 日志持续推回仓库（jobs/logs/），退出码写 jobs/done/。
#
# 启动（仓库根目录）：
#   nohup bash scripts/remote_runner.sh > /tmp/runner.log 2>&1 &
# 停止：
#   touch jobs/STOP   （或 kill 进程）
#
# 要求：本机已配置该仓库的 git push 权限；能访问 github.com。
set -uo pipefail  # 不用 -e：单个任务失败不能杀死 runner

BRANCH=${BRANCH:-claude/fervent-archimedes-7b53bk}
POLL_INTERVAL=${POLL_INTERVAL:-60}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-120}
MAX_LOG_BYTES=$((500 * 1024))

cd "$(dirname "$0")/.."
mkdir -p jobs/queue jobs/logs jobs/done

git_commit_push() {
  git add jobs/
  git -c user.name="gpu-runner" -c user.email="runner@local" \
    commit -qm "$1" 2>/dev/null || return 0
  for delay in 2 4 8 16; do
    git push -q -u origin "$BRANCH" 2>/dev/null && return 0
    git pull -q --rebase origin "$BRANCH" 2>/dev/null || true
    sleep "$delay"
  done
  git push -q -u origin "$BRANCH"
}

snapshot_log() {  # $1=任务名：把完整日志的尾部截入仓库内日志文件
  tail -c "$MAX_LOG_BYTES" "/tmp/job-$1.log" > "jobs/logs/$1.log" 2>/dev/null || true
}

next_job() {
  for j in $(ls jobs/queue/*.sh 2>/dev/null | sort); do
    local name
    name=$(basename "$j" .sh)
    [ -f "jobs/done/$name.exit" ] || { echo "$j"; return; }
  done
}

echo "runner started: branch=$BRANCH poll=${POLL_INTERVAL}s"
while true; do
  [ -f jobs/STOP ] && { echo "STOP file found, exiting"; exit 0; }
  git pull -q --rebase origin "$BRANCH" 2>/dev/null || true

  job=$(next_job)
  if [ -n "${job:-}" ]; then
    name=$(basename "$job" .sh)
    echo "== running $name =="
    git_commit_push "runner: $name started"

    (
      while true; do
        sleep "$SNAPSHOT_INTERVAL"
        snapshot_log "$name"
        git_commit_push "runner: $name progress"
      done
    ) &
    snapshot_pid=$!

    bash "$job" > "/tmp/job-$name.log" 2>&1
    code=$?

    kill "$snapshot_pid" 2>/dev/null
    wait "$snapshot_pid" 2>/dev/null
    snapshot_log "$name"
    echo "$code" > "jobs/done/$name.exit"
    git_commit_push "runner: $name finished exit=$code"
    echo "== $name finished, exit=$code =="
  fi
  sleep "$POLL_INTERVAL"
done
