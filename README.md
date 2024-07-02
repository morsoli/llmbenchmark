# llmbenchmark
国内大模型API性能指标比较 - 深入分析TTFT、TPS等关键指标

## 指标定义

- **TTFT:** 首次生成 token 时间。这指的是模型处理进来的请求并开始输出文本的速度，直接关系到用户界面何时开始更新。数值越低，意味着延迟越低，性能越快。（9.99s 表示请求超时）
- **TPS:** 每秒生成 token 数。这是指模型生成文本的速度，控制着完整的响应在用户界面上显示的速度。数值越高，意味着吞吐量更大，性能更快。
- **Total:** 从请求开始到响应完成的总时间，即最后一个 token 生成的时间。总时间 = 首次生成 token 时间 + 每秒生成 token 数 * token 总数。数值越低，意味着延迟越低，性能越好。（99.99s 表示流式输出过程中超时 ）

## 检测机制

- **连接预热** 为了消除 HTTP 连接建立时的延迟，会先进行一次预热连接。

- **TTFT测量:** 首次生成 token 时间的计时从发起 HTTP 请求开始，到在流式响应中接收到第一个 token 时结束。

- **输出token数:** 输入指令统一为【重复内容```提供API搬家服务的大模型们```，禁止额外输出】，由于各类模型指令遵循能力不一样，输出内容略有差异，且 token 计算方式也不一样，此数值以单次测试结果为准。

- **三次尝试，择优记录:** 对于每个服务提供商，会进行三次独立的推理测试，然后选择最佳的结果（以排除由于排队等造成的异常值）。


## 数据详情
- **西南1区**（[数据详情](https://raw.githubusercontent.com/morsoli/llmbenchmark/main/website/public/cdr.json)）

- **华北1区**（[数据详情](https://raw.githubusercontent.com/morsoli/llmbenchmark/main/website/public/bjr.json)）


## 其他
欢迎关注 [莫尔索随笔](https://liduos.com/)