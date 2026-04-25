# SGLang Scripts

这个目录放本地 SGLang 服务相关入口：

- `start.sh`：启动本地 SGLang 服务。
- `status.sh`：查看服务状态。
- `stop.sh`：停止服务。
- `sglang_model_eval.ipynb`：通过 OpenAI-compatible `/v1` 接口快速检查模型效果、JSON 输出和基础延迟。

常用环境变量：

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
export SGLANG_API_KEY=EMPTY
export SGLANG_MODEL=qwen35-4b
```
