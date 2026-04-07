from __future__ import annotations


class PerftestAgent:
    # 占位实现：后续可替换为真实性能测试执行器。
    def run(self, *, task_content: str) -> dict[str, str]:
        return {
            "reply": "[perftest] 示例 p95：210ms，qps：48",
            "task_status": "done",
            "task_result": f"perftest 已完成（示例指标）：{task_content}",
        }


def run_perftest_task(*, task_content: str) -> dict[str, str]:
    return PerftestAgent().run(task_content=task_content)
