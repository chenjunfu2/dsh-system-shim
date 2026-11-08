# dsh-system-shim

Some third-party DeepSeek-compatible gateways silently drop the Anthropic
`system` field. This is a small standard-library reverse proxy that moves the
system prompt into the head of the first user message, where gateways reliably
forward it.

No dependencies. No build step. Streaming is preserved.

MIT License.

---

## 实测：网关确实会丢弃 system

用同一个网关，同一个模型（`cn:deepseek-v4.1-flash`），同一段 5766 字符的 system 提示，末尾带一条无歧义硬指令（"无论用户问什么，整个回复必须恰好是 BANANA 这个词"），并带上 `tools` 与 `stream: true` —— 也就是 DSH 真实请求的形态：

| 配置 | 模型回复 | 结论 |
|---|---|---|
| A. 直连网关，`system` 字段 | **空** | 指令从未到达模型 |
| B. 直连网关，system 拼进首条 user | `BANANA` | 到达并被遵守 |
| C. 经本代理 | `BANANA` | 到达并被遵守 |

对比的干扰项已排除：同一段 system 内容在**短请求**（无 tools、system 很短）下直连也能生效，说明不是内容问题，而是网关在较长或带 tools 的请求里丢了该字段。如果你只做短请求测试，会看不到这个缺陷。

## 它做什么

一个改写，仅此一个：

```
system: "..."            →   messages[0].content[0] = {type: "text", text: "..."}
```

其余一律原样转发。默认还会**删除** `system` 字段——留着会让两者都生效的网关上出现重复指令，而重复指令比任一单独放置都更糟。需要保留时加 `--keep-system`。

安全的边界：

- 只改 `POST` 且路径以 `/v1/messages`、`/v1/messages/count_tokens` 或 `/messages` 结尾的请求。`/v1/models` 等不受影响。
- 请求体不是 JSON 时原样转发，不猜测。
- `system` 为空或缺失时**完全不改**请求体，连字段都不动。
- `messages` 为空时新建一条 user 消息；首条不是 `user` 时插入一条新 user 消息，而不是把 system 塞进 assistant 轮次。
- 已有 `content` 是字符串时升级为 block 数组，原有块顺序不变，且不修改调用方传入的列表。
- 多块 `system` 会被拼成一个块（`\n\n` 连接），保证头部始终是单个前缀；非文本块（如图片）丢弃，`cache_control` 标记丢弃（它描述的是已不存在的 system 段缓存）。

## 流式：不缓冲

这一点单独测过。代理用 `read1()` 逐块读取，每块立刻以 chunked 编码写出并 `flush()`，因此 token 逐个到达，而不是等全部生成完再一次性吐出。

测试断言的是**到达时序**而非内容——一个把整个响应缓冲后才返回的代理，内容 diff 看起来完全正确，但在 UI 上不可用。mock 上游在每个事件间插入 120ms 间隔，5 个事件要求首末到达间隔大于一半预期值；缓冲的实现会得到间隔约 0 而失败。

## 用法

```sh
python system_shim.py --upstream http://host:7863/anthropic/ --port 8787
```

然后把客户端的 base URL 指到代理根：

```yaml
baseURL: http://127.0.0.1:8787/
```

代理会把请求路径追加到上游 base path 之后：`/v1/messages` → `<upstream>/v1/messages`。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--upstream` | 必填 | 网关 base URL，如 `http://host:7863/anthropic/` |
| `--listen` | `127.0.0.1` | 监听地址 |
| `--port` | `8787` | 监听端口 |
| `--keep-system` | 关 | 保留原 `system` 字段（会重复） |
| `--record DIR` | 无 | 把改写前/后的请求体存盘，用于核验 |
| `--verbose` | 关 | 记录每个请求与改写情况 |
| `--probe` | 关 | 打印一个示例请求的改写结果后退出，不启服务 |

### 核验网关是否真的丢弃 system

`--record` 能证明**代理**改写正确，但证明不了**网关**是否接受原生字段。要测网关，用一段足够长的 system 加一条不容易被忽略的固定指令，并带上 tools，比较直连与经代理两种形态的回复差异：

```sh
# 若直连返回空/无关内容，经代理返回预期内容，就是网关丢了 system
```

## 测试

```sh
python -m unittest discover -s . -p "test_*.py"
```

- `test_system_shim.py` — 19 个纯函数测试：`system` 为字符串/块数组/缺失/空白、首条为 assistant、`messages` 为空、保留原字段、其他顶层字段不被改动、非 dict 输入、路径匹配。
- `test_integration_system_shim.py` — 11 个集成测试：真实启动 mock 上游与代理，验证改写到达上游、路径与 query 拼接、`x-api-key` 与 `anthropic-version` 透传、非 messages 路径不改写、流式逐块到达（防缓冲）、多字节中文不被破坏。

两者都在进程内用线程起服务，不 spawn 子进程，不会残留后台进程。

## 已知限制

- **这是绕过网关缺陷的权宜之计，不是协议特性。** 网关若哪天开始尊重 `system`，把它挪走仍然有效，只是不再必要；届时更该做的是换网关或提 issue。
- **prompt caching 会受影响。** 原本在 system 段的 `cache_control` 断点被丢弃，system 文本并入 user 消息后处于不同的缓存前缀位置。如果网关支持缓存且你依赖它，需要重新评估。
- **只处理前缀，不处理语义。** system 与 user 的相对权重在协议上确实不同；把指令写进 user 消息是退让，不是等价替换。对"必须遵守"类规则足够，对依赖 system 特权语义的用法不一定。
- **HTTP/1.1 chunked 与已知长度二选一。** 响应固定用 chunked，因为长度在流式时未知；这要求客户端支持 chunked（DSH 的 Anthropic 适配器可以）。
- **无 TLS 终止。** 代理说 HTTP，`--upstream` 可为 HTTPS。客户端到代理之间是本机明文，不要跨机暴露。

## 与 dsh-system-prompt-inject 的关系

两者解决的是不同层的两个问题，可同时使用：

- [dsh-system-prompt-inject](https://github.com/chenjunfu2/dsh-system-prompt-inject) 解决**放置**问题：DSH 里 `AGENTS.md` 是 user 段、preset 会遮蔽 `personaPrefix`，所以需要一个不同名字的 system 段才能把规则送上 system 层级。
- 本代理解决**传输**问题：规则到了 system 层，但网关把它丢了。

只装前者而网关丢弃 system，表现为"注入成功但模型完全不理"——system 里确实有那段文字，模型却从没收到。这正是本仓库存在的理由。
