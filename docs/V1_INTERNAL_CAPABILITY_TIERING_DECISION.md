# V1-M2 Internal Capability Tiering Decision

## 1. Status and decision

本文档冻结从 `V1-M1-C1` 到 `V1-M1-C2`、再到 `V1-M2` 的模型内路由路线。已实现的
`V1-M1-C1` `AuthenticatedR2` 保留为最小 `DENY/protected` reference：它只隔离验证、唯一协调器、
冻结 R2 语义等价和拒绝零 R2 调用。`V1-M1-C2` 是第一个正式 public/protected 二专家硬路由：
显式 public entry 执行公共专家 `E0`，受保护 entry 只有在协调器提交 `PROTECTED` 后执行冻结
受保护专家 `E1`。

`V1-M2` 只在 C2 闭合后把单个 `E1` 推广为多个受保护专家、由可信本地策略提交不可扩大的权限
掩码，并让任务 router 只在已授权集合内选择。C1 是内部 reference/evaluator，不作为长期对外能力；
C2 是首个对外有意义的模型内能力分层实验。三者都不是对 V0、V1-prep、V1-P2 协议、V1-C1
verifier 或 V1-M1 的重命名、替换或弱化。所有 V1-P2 credential、A3-v2 transcript、registry 和
neural verifier 保持原有 protocol identifier、canonical encoding 和接受集合。

本决定研究的是受控黑盒服务假设下的模型内条件化路由，不是白盒不可绕过、模型权重防篡改、
TEE、安全启动、远程证明或任意模型变换后的安全保证。

## 2. Threat model and claim boundary

C2/M2 延续 `SECURITY.md` 的受控黑盒推理服务假设：请求方只能使用公开组合入口，不能读取或修改
R2 权重、推理代码、Gate Layer、协调器或深层 suffix，也不能获得 protected logits 或中间特征。
攻击者可以提交任意业务输入与 credential bytes，进行畸形编码、类型混淆、边界值、replay、
tamper 和并发 response 尝试。

C2/M2 可以主张：

- 对 canonical credential，固定 V1-C1 neural verifier 的接受结论保持既有
  `V_nn(a) = 1 -> V_ref(a) = 1` 证明；
- 认证成功时，protected 路径与冻结 R2 业务语义等价；
- public 或 pre-commit reject 路径不执行 protected suffix，不释放 protected logits、features 或
  可转移 protected capability；
- credential 由显式 Module-SIS commitment/challenge/response transcript 表示，并绑定 canonical
  input digest、model/profile、identity、scope、nonce 与 expiry。

C2/M2 不可以主张：

- V1 toy/conformance profile 的不可伪造性、具体参数安全、主动冒充安全或生产认证安全；
- 进程所有者、白盒权重持有者或可改写推理代码的攻击者不能删除 Gate 或直接调用 suffix；
- 任意 pruning、quantization、fine-tuning、export 或其他模型变换仍保留上述性质；
- public coarse capability 在信息论意义上不泄露任何有关 protected model 的信息。

## 3. Frozen architecture contract

设已验收 R2 为 `f_theta`。C2 必须以不改变 R2 参数、module 顺序或 eval semantics 的方式选择
一个拓扑切分：

```text
f_theta = d_theta o s_theta
```

其中 `s_theta` 是共享 frozen prefix，`d_theta` 是共享 frozen protected suffix。新建的 public head
`g_psi` 只消费 `s_theta` 的内部输出；`psi` 与 `theta` 分离，训练 public head 时 `theta` 必须冻结。
C2 的外部可观测行为为：

```text
F_public(x) = g_psi(s_theta(P(x)))
F_protected(x, a) = d_theta(s_theta(P(x)))  only after C commits allow for a
F_protected(x, a) = DENY                    otherwise
```

`P` 是由 V1-M1 input adapter 产生的 canonical uint8 snapshot 和固定 preprocessing。认证分支只
处理分离的 canonical transcript，不把 credential 写入图像、特征、prompt 或普通业务输入。不得使用
`gate * logits`、输出遮蔽、soft routing、learned gate 或图像 Secret Trigger 模拟硬路由。

组合对象必须注册固定 `V_phi`、R2 prefix、R2 suffix、public head 和内部协调器。`V_phi` 只产生
evidence；`C` 仍是唯一提交 `PUBLIC`、`PROTECTED` 或 `DENY` 结果的组件。protected challenge 一旦
进入验证但失败，必须返回 `DENY`，不能 fallback 到 public；显式 public entry 仅能产生 public capability。

这里的固定 `V_phi` 由 exact Module-SIS relation、canonical input domain 和可信公开 profile
确定性编译，不从 credential 样本学习，也不保存 prover secret、签名私钥或静态 Secret Trigger。
NNAES 只作为固定密码神经图的构造类比，不能让 C2/M2 继承其 key-specific 网络布局或任何域外安全
结论。所有非规范连续输入必须在 parser/adapter 边界拒绝，不能交给 `V_phi` 插值判定。

M2 在此二专家契约上增加受保护专家集合 `E_1...E_k`。canonical claims 只提供由 transcript 绑定的
identity/scope 等证据字段，唯一协调器依据可信本地 policy/registry 提交不可变 `RouteContext` 与
`allowed_mask`；验证器、请求方和普通 router 均不能直接铸造权限掩码。任务得分只用于已授权集合内
的选择，未授权专家 forward 必须保持零调用。若受保护请求的授权集合为空、scope 不匹配或路由失败，
结果为 `DENY`，不能改走 `E0` 或其他受保护专家。

## 4. Public capability and cut selection

C2 的第一项 public capability 固定为 CIFAR-100 的 20-class coarse superclass prediction。public
响应不返回 100-class logits、R2 classifier output、prefix feature 或任何可重标记为 protected 的 token。

切分位置不得在看到 test metrics 后临时选择。实现前必须在本地决策中列出有限候选 prefix 集合、
public-head topology、训练种子、validation-only selection rule、public capability acceptance threshold
和禁止调整项。R2 的 accepted state 不重训、不微调、不以 public task metrics 为由更换；只有独立
`g_psi` 可在冻结 input/data protocol 下训练。

## 5. Proof and test obligations

C2 必须分别验收，不能以单一 accuracy 指标替代：

1. **Neural relation soundness:** 使用既有 V1-C1 全 canonical input `V_nn == V_ref` 证明，以及
   C2 route 对 malformed/noncanonical transcript 的 fail-closed 测试。
2. **Protected semantic preservation:** 在 accepted R2 test artifact 的评估范围内，C2/M2 protected
   logits 与直接冻结 R2 logits 逐元素相等；prediction digest 必须相同。
3. **Routing isolation:** public、parse reject、relation reject、expiry、replay、tamper、concurrent
   duplicate response、route confusion 和内部 pre-commit error 均为零 protected-suffix calls，并不释放
   protected logits/features。
4. **Public-path capability:** public path 只调用 prefix/public head，输出仅属于冻结 coarse label domain；
   public head 的 validation/test 指标、parameter count、latency 和调用计数独立记录。
5. **Input binding:** 非标准实数、错误 dtype/shape/range、非规范 credential 和业务输入 mutation 在
   parser/adapter 层拒绝；对已签发 transcript 的任何 canonical image byte 修改必须导致 binding reject。

M2 还必须单独验收 route soundness：任何 `j > 0` 的专家实际执行都蕴含 reference verifier 接受、
协调器已经提交 `PROTECTED`、`j` 属于可信 policy 生成的 scope mask，且该 mask 不大于 credential
绑定 scope。full-scope credential 与完整 MoE reference 对照；restricted-scope credential 只与同一
授权集合上的 constrained-router reference 对照，不能错误要求它等于未受约束的完整 MoE。

## 6. Transformation research boundary

C2/M2 首期只支持已冻结的 Python/PyTorch module graph 与环境。state reload、module refactor、quantization、
pruning、fine-tuning、export 或其他变换默认使 M2 certificate 失效，必须拒绝部署或作为新 profile
重新执行 relation、semantic-preservation 和 routing-isolation 验收。

后续可以新增受限的 `V1-M2-R1` transform study，分别报告变换前后是否仍通过上述验收；该研究仅评估
受信部署变换，不把攻击者任意改写模型的能力纳入安全主张。没有制品完整性机制时，任何模型内 Gate
都不能阻止该攻击者直接移除 Gate 或调用 suffix。

## 7. Implementation increments and resources

| Increment | Resource | Required outcome |
| --- | --- | --- |
| Route decision freeze | `LOCAL_OK` | 本文档、工作日志、研究设计和安全边界一致；治理检查通过 |
| M1-C1 local `AuthenticatedR2` reference | `LOCAL_OK` | V1-C1/A3-v2 控制冻结全 R2；conformance allow equivalence 与 reject zero-call 测试通过 |
| M1-C1 accepted-R2 reference report | `SERVER_REQUIRED` | 10,000-image direct/gated R2 logits 等价、拒绝隔离与分段 latency 通过；不训练模型 |
| M1-C2 split/composition | `LOCAL_OK` | prefix/suffix 重组的 direct-R2 logits 等价；显式 public/protected/deny 路径和隔离测试通过 |
| M1-C2 public-head training | `SERVER_REQUIRED` | 在冻结 R2、数据和预注册训练规则下训练/选择 public head；不得重训 R2 |
| M1-C2 accepted-state report | `SERVER_REQUIRED` | 10,000-image public/protected/deny 报告、分段 latency 和 artifact manifest 验收 |
| M2 multi-expert constrained routing | `LOCAL_OK` before formal model evaluation | 固定 scope-to-mask policy、受约束 router、空集合/route confusion 与逐专家零调用测试 |
| M2-R1 transform evaluation | `SERVER_REQUIRED` only when formal data/model evaluation begins | 逐变换重新验收或明确失效；不扩大为白盒保证 |

首次训练 public head、生成正式 C2/M2 artifact 或运行正式 GPU 性能测量前，
`PROJECT_WORKLOG.md` 的唯一下一步必须先改为 `SERVER_REQUIRED` 并通知项目负责人。

## 8. Required implementation boundaries

- 新模块使用 V1-local identifier、registry、entry points、response schema、tests 和 artifact namespace；
  不改变或接受 V0/A2/A3-v1/A4/V2 wire routes。
- public/protected path 共享的仅是同一个冻结 R2 prefix 参数；不得复制、训练或写出含 R2 state 的新 artifact。
- verifier、R2 prefix/suffix、public head 与 coordinator 不持有客户端 secret；V1-P2 fixture secret 只留在
  临时测试/experiment 上下文。
- 静态 Secret Trigger/密码字符串若未来作为实验对照，必须使用独立 identifier、parser、entry 和
  artifact namespace，明确其 bearer/replay 限制；不得接入 C2/M2 主路线或形成认证 fallback。
- public capability 不能升级为 protected capability；protected failure 不能降级为 public；任何 request
  字段都不能选择 verifier、profile、threshold、cut、head 或 authorization decision。
- 研究报告必须把 C1 最小 reference、C2 二专家硬路由、M2 多专家受约束路由、public head quality、
  V1 neural relation soundness、协议安全假设和不支持的白盒/transform 场景分开叙述。
