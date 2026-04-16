# 近期更新对齐

范围：2026-04-14 至 2026-04-16

## 时间线

| 日期 | 提交 | 主题 | 影响面 |
|---|---|---|---|
| 2026-04-16 | - | 引入 pyskill 模式（skill-mode）与进程巡检回填 | skill_tool 支持非阻塞进程派发；pre-reply 巡检可对死进程/超时自动 failed 收敛；支持重启后 running 任务兜底 |
| 2026-04-16 | - | Skill 体系切换为 ClaudeCode 风格 | 统一 `SKILL.md + scripts + allowed-tools`，新增 `paths.skills_root`，`web_search` 下沉到 `skill_tool` |
| 2026-04-16 | - | docs 对齐：skill 插件化架构与扩展规范 | 文档与当前实现一致，新增 `docs/skills.md`，降低新增 skill 的认知成本 |
| 2026-04-15 | `d889c46` | 技能元数据驱动执行 | executor 自动扫描 `SKILL.md`，注入 `name/description/when_to_use/path` 供模型选择 |
| 2026-04-15 | `e11424c` | 收敛 executor 失败路径与约束 | 减少失败污染与无效重试，提高执行稳定性 |
| 2026-04-15 | `b0beb5d` | 引入 agent memory；environment 增加视图级压缩 | 降低 step 上下文拼接噪声与超窗风险；不改变落盘 schema |
| 2026-04-15 | `19f9def` | 状态追问快捷汇总；controller observe 参数 schema 收敛 | 减少“现在怎么样了”场景下的无效 observe 重试和 `read` 参数错误 |
| 2026-04-15 | `b4d7de1` | workflow 非阻塞回填；交互模式复用 environment | 长任务不阻塞当前轮；异步完成后可在当前 round 回填 `pyskill_task` |
| 2026-04-15 | `c036700` | `normal` 统一更名为 `executor` | task type 命名统一，路由语义更清晰 |
| 2026-04-15 | `a7c4926` | normal 体系重构并收敛到 executor 语义 | 旧命名兼容路径清理，简化理解成本 |
| 2026-04-15 | `c3a7846` | 启动时检测 sglang，不可用回退 aliyun | 降低本地依赖不可用导致的启动失败 |
| 2026-04-15 | `3d5c124` | test agent 增加固定 5 秒 mock sleep | 模拟长任务，验证 running/回填/追问链路 |
| 2026-04-14 | `d53cae7` | `previous_failed_track` 支持跨 round 定位最近失败任务 | 失败诊断和重试在多轮对话中更稳定 |

## 设计亮点（对应近期改动）

1. 异步与对话解耦：执行耗时与用户交互拆开，体验上更像真实任务系统。
2. 回填可追踪：source task 与 `pyskill_task` 通过引用关联，既保留原始意图，也保留执行结果实体。
3. Skill 插件化：executor 通过元数据注入 + 按需 read 正文，支持低侵入扩展。
4. 路由防抖：controller observe 参数与工具使用边界收紧后，失败更可控、可诊断。
5. 环境复用一致：交互式 CLI 的状态连续性与落盘结果一致，调试效率更高。
6. 失败信息可继承：跨 round 的失败轨迹查询降低了“下一轮失忆”问题。
7. 上下文可控：memory 压缩与视图压缩分层，既控 token 又保留持久化兼容性。

## 建议阅读顺序

1. `docs/design.md`：先看编排、分支语义与 skill 注入链路
2. `docs/skills.md`：再看 skill 目录规范、元数据注入与扩展步骤
3. `docs/skills_runtime.md`：再看加载校验、skill_tool 契约与脚本执行细节
4. `docs/environment.md`：再看落盘结构与异步回填口径
5. `docs/agent_memory.md`：最后看 memory 与视图压缩机制
