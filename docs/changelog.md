# 近期更新对齐

范围：2026-04-14 至 2026-04-29

## 时间线

| 日期 | 提交 | 主题 | 影响面 |
|---|---|---|---|
| 2026-04-29 | `a627a36` | docs(post-training): 同步 GRPO DPO 方案入口 | README、overview、data contract、post training、assets README、changelog 同步加入 `GRPO / DPO` 下一阶段链路入口；`preference_admissions` 作为候选回流对象进入文档索引 |
| 2026-04-29 | `ab89763` | docs(post-training): 增加 GRPO DPO 候选方案 | 新增 `grpo_dpo_loop_v1.md`，明确 `SFT warm start -> (GRPO online rollout -> preference_admissions -> DPO) -> ...` 的目标循环、pair schema、验收线和 stale preference 风险 |
| 2026-04-28 | `3cf93fa` | docs(post-training): 增加 GRPO 前后评测协议 | 新增 `grpo_ab_eval_v1.md`，定义固定 holdout paired eval、三维评分、deterministic overrides、acceptance bar 与诊断 runbook；当前实现尚未补齐 compare CLI |
| 2026-04-28 | `b46c006` | docs: 更新 Environment-Runtime 仓库名 | README 与文档口径从旧名称统一切换到 Environment-Runtime |
| 2026-04-28 | `33dfe51` | config: 切换阿里云默认模型 | 默认模型配置切回阿里云路径，降低本地 SGLang 不可用时的启动门槛 |
| 2026-04-27 | `30995f8` | feat(post-training): 补齐 GRPO 诊断与 holdout 评测 | 后训练链路补齐 GRPO 诊断输出和固定 holdout 评测能力，为 checkpoint 前后对比打基础 |
| 2026-04-27 | `18fc00f` | feat(post-training): 使用 GRPO checkpoint 评测 holdout | holdout 评测接入 GRPO checkpoint，支持直接评估 GRPO 后模型输出 |
| 2026-04-27 | `925cae1` | fix(eval): 表格化 GRPO 与 holdout 诊断 | GRPO / holdout 诊断输出改为更可读的表格结构，方便定位 fixed / regressed / bucket 问题 |
| 2026-04-27 | `27f6c0d` | fix(grpo): 收紧 KL 约束参数 | 调整 GRPO KL 约束，降低后训练阶段偏离 SFT 协议地基的风险 |
| 2026-04-27 | `6431faa` | fix(grpo): 修复 SFT 策略接入与 reward 审计 | 修复 GRPO 使用 SFT policy 初始化与 reward audit 记录问题，保证后续诊断能回看 teacher ranking 与 reward 来源 |
| 2026-04-25 | `691a9db` | feat(post-training): 补齐 holdout 预测与评测图表 | 增加 holdout prediction / eval 图表产物，补齐训练后效果查看入口 |
| 2026-04-25 | `64bc595` | feat(train): 补齐单机多卡 SFT 与 GRPO 入口 | 增加本地单机多卡 SFT / GRPO 训练入口，为本地 controller 后训练打通执行路径 |
| 2026-04-25 | `40102e7` | data(train-assets): 添加 round_0001 后训练资产 | 新增 `round_0001` 后训练资产，形成 round 资产主线的首个可执行样例 |
| 2026-04-25 | `e021d4a` | refactor(train-mainline): 收口 admissions 生命周期与 GRPO 资产契约 | 收口 `teacher_queue / sft_admissions` 生命周期和 GRPO 资产契约，形成当前 SFT 回流主线的实现基础 |
| 2026-04-25 | `9789d8c` | docs(train-docs): 同步 round 主线与队列入口说明 | 训练文档同步 round 资产主线、teacher queue 与 admissions 入口 |
| 2026-04-24 | `89248eb` | run: 移除 streamlit 入口并只保留 CLI | 删除 `scripts/run/streamlit_app.py`，移除 `requirements.txt` 中的 `streamlit` 依赖，README 的运行方式与目录说明同步收口到 CLI / case 入口 |
| 2026-04-24 | `53b744f` | readme: 优化示例 task type 文案 | README 中反复出现的 `functest / accutest / perftest` 统一解释为当前仓的示例 task family，避免把框架定位误读成测试专用实现 |
| 2026-04-23 | `30ecc39` | docs: 收口 README 与训练文档口径 | 根 README 与 `src/task_router_graph_train/docs/` 全面改写到当前闭环口径，统一为 manifest 安全输入、三路 teacher、badcase 回流、controller regression 与 repo-relative 路径输出 |
| 2026-04-23 | `2f99f8e` | security(path): 脱敏绝对路径输出并收口训练报告路径 | 运行入口、训练报告、feedback manifest、regression/holdout run manifest 默认输出 repo-relative 路径，降低跨机器 diff 噪声与本机路径泄露风险 |
| 2026-04-23 | `75e24be` | tests: 覆盖 badcase 回流与 manifest 输入校验 | 新增 badcase feedback loop、manifest 安全输入、结构约束相关测试，保证 `train_sft` / `train_grpo` 默认口径稳定 |
| 2026-04-23 | `01ed73e` | train-entry: 用 manifest 收口 SFT/GRPO 安全输入 | `train_controller_sft(...)` 与 `train_controller_grpo(...)` 默认优先消费 `asset_manifest` / `run_dir`；直接路径 override 仅在显式开启 `allow_unsafe_path_input` 时可用 |
| 2026-04-23 | `f6a3e45` | controller-regression: 接入独立 teacher 判分与覆盖率面板 | 新增 `evaluate_controller_regression(...)` 与对应 CLI，支持独立 `regression_judge`、evidence rows、按 bucket 汇总指标和 coverage 面板 |
| 2026-04-23 | `3b12394` | badcase-feedback: 新增回流资产构建与运行清单 | 新增 `build_feedback_assets(...)`、`harvest_failed_badcases(...)`、run-scoped `feedback_manifest.json`、feedback 资产索引与 failed badcase 回流闭环 |
| 2026-04-23 | `5eb2a63` | teacher-config: 拆分 reward/reference/regression 三路 teacher 配置 | controller online config 中的 teacher 拆为 `reward_judge / reference_generator / regression_judge`，GRPO、feedback、regression 三条路径各自消费独立 teacher 角色 |
| 2026-04-23 | `c196d76` | docs(task_router_graph_train): 补充 verl GRPO 训练说明 | `task_router_graph_train` 正式文档开始收口到当前单步 controller GRPO on verl 的实现边界 |
| 2026-04-22 | `4f7e83f` | grpo-pipeline: 新增 Teacher-RM controller 后训练与回流闭环 | controller 后训练首次打通 `teacher ranking -> preference rows -> update` 主线，并把回流闭环正式纳入训练视角 |
| 2026-04-22 | `88412b9` | sft-schema: 删除 step 字段并收敛 metadata（保留 terminal） | SFT record / example 结构进一步瘦身；`metadata` 聚焦 `terminal` 等当前真正有用的结构信息 |
| 2026-04-22 | `c99a9d9` | sft-contract: 移除 reward spec 字段（manifest + records） | SFT 资产与 `reward_spec_id` 脱钩，reward spec 口径开始收口到 RL / Eval 路径，减少训练输入混杂字段 |
| 2026-04-22 | `e442182` | refactor(sft-assets): 对齐 teacher_source 与构建链路说明 | teacher_source、records、examples 的构建说明与实际产物重新对齐，降低从数据真源到训练入口的理解成本 |
| 2026-04-22 | `77aa573` | feat(reply): 强制显式提及 pyskill 完成事件 | reply 输出更稳定地暴露 pyskill 完成事实，减少“任务已经完成但回复没提到”的感知断层 |
| 2026-04-22 | `69ff9cb` | fix(graph): scope status shortcut by target test type and configurable mock async sleep | 状态追问 shortcut 按目标 task type 收紧，mock async sleep 变为可配置，减少快捷汇总误触发 |
| 2026-04-22 | `74c03bf` | docs(changelog): record ENVIRONMENT_JSON migration and skill updates | 首次把 `ENVIRONMENT_JSON` 迁移和 skill 体系更新写入 changelog，形成从 contract 变化到能力变化的显式记录 |
| 2026-04-22 | `c5ebc57` | feat(time-range-info): refresh skill workflow and add training schemas | `time_range_info` skill 的 workflow、训练 schema 与文档同步刷新，为后续专题训练与策略化实验打底 |
| 2026-04-22 | `d8fb1dc` | refactor(contract): rename TASKS_JSON to ENVIRONMENT_JSON across router pipeline | router pipeline 的关键输入名统一迁移到 `ENVIRONMENT_JSON`，让训练态 state 与运行时语义更一致 |
| 2026-04-17 | - | 训练/运行拆分：RL 真源迁入 `task_router_graph_train` | 训练实现从 `task_router_graph` 运行时包移出；新增 `src/task_router_graph_train/` 承接 docs、assets、CLI、reward spec 与离线评测；项目根不再作为 RL v1 正式入口 |
| 2026-04-16 | - | 数据口径重构：统一手工 holdout 与历史归档 | 后训练评测不再区分 mock/real；手工样本与正式 holdout 由训练模块内部资产目录承接；过时 `cases/environments/rl/mock` 迁移至 `data/archive_legacy/2026-04` |
| 2026-04-16 | - | 后训练目标改为问题修复导向（真实轨迹优先） | RL 目标转向 formal `environment/task` 语义建模与修复决策；文档与评测规范迁入 `src/task_router_graph_train/docs/` |
| 2026-04-16 | - | time_range_info 升级为 Agentic CRAG + 混合检索 | `time_range_info worker graph` 改为 `bootstrap -> hybrid retrieve -> LLM grader -> rewrite loop -> synthesize`；主 graph 本体不变，pyskill 负责异步触发与回填；新增 embedding 配置入口并增强静态扫描约束 |
| 2026-04-16 | - | docs: PySkill 设计稿补充“落地对照 + TODO” | 明确已落地亮点（非阻塞派发、pre-reply 收敛、run_id 幂等）与未落地能力（heartbeat/step 可视化等） |
| 2026-04-16 | - | 引入 pyskill 模式（skill-mode）与进程巡检回填 | `skill_tool` 支持非阻塞进程派发；pre-reply 巡检可对死进程/超时自动 failed 收敛；支持重启后 running 任务兜底 |
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

## 本阶段重点变化

1. 训练闭环从“分散入口”收口到 manifest / round 资产主线。
2. 当前已实现链路是 `manual_protocol_v1 -> SFT -> GRPO -> teacher_queue -> annotate_queue -> sft_admissions`。
3. badcase 直接回流到下一轮 SFT 已被标记为过渡方向；后续重点转向 `GRPO online rollout -> preference_admissions -> DPO` 循环，其中 teacher gold answer 与 bad output 会组成偏好样本。
4. `GRPO / DPO` 下一阶段方案已形成独立文档，核心是保留 `chosen / rejected` pair，而不是把 badcase 压平成单条 SFT reference。
5. teacher 职责当前按正式链路收口为 reward ranking、holdout 语义判等与 badcase admission 三类评判；下一阶段会增加 gold answer / preference admission 口径。
6. 固定 holdout evaluation 已补齐单 checkpoint 评测和 GRPO 诊断；paired comparison 指标仍是目标协议，compare CLI 尚未落地。
7. 训练、评测、CLI 报告中的路径统一脱敏成 repo-relative 形式。
8. `TASKS_JSON -> ENVIRONMENT_JSON` 的输入命名迁移已经落地，训练态 state 与运行时 contract 更一致。
9. 根 README 对外定位改成通用工程场景，`functest / accutest / perftest` 明确标为当前内置示例 task family。
10. 运行入口进一步收紧，当前仅保留 CLI / case 方式，移除了 streamlit 实现与依赖。

## 建议阅读顺序

1. `README.md`
   - 先看运行时框架定位、CLI 入口和 skill / pyskill 主流程
2. `src/task_router_graph_train/README.md`
   - 再看训练模块目前真实打通到哪一步
3. `src/task_router_graph_train/docs/overview.md`
   - 看训练闭环总图与输入输出约定
4. `src/task_router_graph_train/docs/data_contract.md`
   - 看 manual protocol、round manifest、teacher_queue、sft_admissions 的当前契约，以及 preference admissions 的下一阶段契约
5. `src/task_router_graph_train/docs/controller_grpo_reward_spec.md`
   - 看 GRPO reward、hard gate 与 teacher ranking 的当前规则
6. `src/task_router_graph_train/docs/grpo_dpo_loop_v1.md`
   - 看 `SFT -> GRPO -> DPO` 下一阶段方案
7. `src/task_router_graph_train/docs/post_training_v1.md`
   - 最后看当前已实现链路、目标评测协议与 DPO 演进边界

## 下一步

优先实现训练侧新回流主线：

```text
GRPO online rollout -> teacher gold answer / bad output pair -> DPO -> GRPO online rollout -> ...
```

目标：

- 保留当前 policy bad output 作为 rejected
- 生成 teacher gold answer / chosen response
- 写入 `preference_admissions`，形成 DPO 可消费的 pair 数据
- 将 `sft_admissions` 收窄为 warm start 补充与协议修补
