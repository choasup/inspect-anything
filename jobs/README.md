# jobs/ — 异步任务队列

GPU 机器入站不可达、云端沙箱出站受限时，用仓库本身作为双方都可达的指挥信道：

```
jobs/queue/NNN-名称.sh   # 任务脚本，按文件名顺序执行（由会话侧下发）
jobs/logs/NNN-名称.log   # 执行日志尾部（runner 持续推回，上限 500KB）
jobs/done/NNN-名称.exit  # 退出码（存在即视为已执行，不会重跑）
jobs/STOP                # 创建此文件可让 runner 优雅退出
```

## GPU 机器侧（一次性）

```bash
git clone <仓库地址> -b claude/fervent-archimedes-7b53bk
cd inspect-anything
nohup bash scripts/remote_runner.sh > /tmp/runner.log 2>&1 &
```

要求：本机有该仓库的 push 权限、能访问 github.com
（先用 `git ls-remote origin` 验证）。

## 行为约定

- 任务在仓库根目录执行，输出全量日志在机器的 `/tmp/job-<名称>.log`，
  仓库内只保留尾部 500KB（每 2 分钟推一次进度快照）；
- 任务失败（非零退出码）不影响后续新任务，runner 不会自动重跑——
  重跑 = 删除对应的 `jobs/done/*.exit` 后推送；
- 长任务（训练）期间可以随时 `git pull` 看 logs 下的进度。
