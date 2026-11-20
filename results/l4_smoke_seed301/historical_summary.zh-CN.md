# L4 Astra 订阅通道：seed 301 单试次

- 任务：`region_place`，异常检查点 `TARGET_POSITION_CHANGED`；使用非特权 RGB-D、K=3 和既有 seed 301 检查点。
- 模型：`gpt-6-astra`，本机已登录的 ChatGPT 订阅通道；无 OpenRouter/API Key 回退。初态指纹 `759a63491134c6a683db26397e80d45498196ad9ab7ae09c6a6d4f87601e76f5`，与此前 L3、原生 CAPX 的 seed 301 指纹相同。
- 有效试次：1/1 物理成功。`trial_end` 为 `success=true`、`task_completed=true`；终态物体在目标区域内且稳定，夹爪已张开，未抓持物体。模型提交 `recovery_graph_v1`，本地编译为受控的重新观测、重抓、重新放置步骤。该初始模型请求附带图像数为 0；RGB-D 观测由后续受控执行步骤使用，不能称为模型看图规划。
- 有效试次资源：1 次模型请求；输入 10,951、输出 101、合计 11,052 tokens；模型响应 12.60 秒；试次总耗时 37.31 秒。两个视频均可解码。
- 预检中止：首次请求返回了非预期的说明对象，未进入有效试次；另消耗输入 10,795、输出 187、合计 10,982 tokens，响应 17.90 秒。因此本次实际订阅通道调用合计 2 次、22,034 tokens。账户额度快照未成功取得，不能把 tokens 换算为精确订阅额度扣减。
- 结果范围：这里只证明一个 L4 单试次可通过当前订阅通道执行，**不能**与先前使用 Gemini 的 L3/原生 CAPX 结果作同模型性能比较。自动生成的 `rerun/report.md` 是三组通用模板；其中的 `0/0` 配对栏不是失败率。

证据：`outputs/structural_unknown_ab/astra_subscription_1seed/model_calls/`；`outputs/structural_unknown_ab/astra_subscription_1seed/rerun/runs/gpt-6-astra/l4/`。
