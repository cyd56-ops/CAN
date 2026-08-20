# V1-M1-C2 and V1-M2 Internal Capability Tiering Decision

## 1. Status and decision

本文档冻结从 `V1-M1-C1` 到 `V1-M1-C2`、再到 `V1-M2` 的模型内路由路线。已实现的
`V1-M1-C1` `AuthenticatedR2` 保留为最小 `DENY/protected` reference：它只隔离验证、唯一协调器、
冻结 R2 语义等价和拒绝零 R2 调用。`V1-M1-C2` 是第一个正式 public/protected 二专家硬路由：
显式 public entry 执行公共专家 `E0`，受保护 entry 只有在协调器提交 `PROTECTED` 后执行冻结
受保护专家 `E1`。

`V1-M2` 只在 C2 闭合后把单个 `E1` 推广为多个受保护专家、由可信本地策略提交不可扩大的权限
掩码，并让任务 router 只在已授权集合内选择。C1 是内部 reference/evaluator，不作为长期对外能力；
C2 是首个对外有意义的模型内能力分层实验。三者都不是对 V0、V1-prep、V1-P2 协议、V1-C1
verifier 或 V1-M1 的重命名、替换或弱化。所有 V1-P2 credential relation、A3-v2 canonical
message/commitment/challenge/response/transcript、registry、input profile 和 V1-C1 neural verifier
保持原有 protocol identifier、wire encoding 和接受集合。

本决定研究的是受控黑盒服务假设下的模型内条件化路由，不提高底层格协议的密码安全性，也不是白盒
不可绕过、模型权重防篡改、TEE、安全启动、远程证明、生产访问控制或任意模型变换后的安全保证。

## 2. Threat model and claim boundary

C2/M2 延续 `SECURITY.md` 的受控黑盒推理服务假设：模型、推理代码、canonical parser、协调器、
registry、hard dispatcher 和已加载 artifact 可信；请求方只能使用规定的组合入口，不能读取或修改
R2 权重、推理代码、Gate Layer、协调器或深层 suffix，也不能获得 protected logits 或中间特征。
攻击者可以提交任意业务输入与 credential bytes，进行畸形编码、类型混淆、边界值、replay、tamper、
跨 route 尝试和并发 response 尝试。

C2/M2 可以主张：

- 对 canonical credential，固定 V1-C1 neural verifier 的接受结论保持既有
  `V_nn(a) = 1 -> V_ref(a) = 1` 证明；
- 认证成功时，protected split path 与冻结 R2 业务语义等价；
- public 或 pre-execution deny 路径不执行 protected suffix，不释放 protected logits、features 或
  可转移 protected capability；
- credential 由显式 Module-SIS commitment/challenge/response transcript 表示，并绑定 canonical
  input digest、model/profile、identity、scope、nonce 与 expiry；
- 在规定组合入口、固定 artifact 和可信 dispatcher 内，public/protected/deny 的实际执行分支与已提交
  route decision 一致。

C2/M2 不可以主张：

- V1 toy/conformance profile 的不可伪造性、具体参数安全、主动冒充安全或生产认证安全；
- 进程所有者、白盒权重持有者或可改写推理代码的攻击者不能删除 Gate 或直接调用 suffix；
- 任意 pruning、quantization、fine-tuning、export 或其他模型变换仍保留上述性质；
- public coarse capability 在信息论意义上隐藏 protected fine-label 语义；
- 单机计数或性能结果能够外推为跨设备、分布式或生产系统保证。

## 3. Frozen architecture and entry contract

设已验收 R2 为 `f_theta`。C2 必须以不改变 R2 参数、module 顺序或 eval semantics 的方式选择一个
拓扑切分：

```text
f_theta = d_theta o s_theta
```

其中 `s_theta` 是共享 frozen prefix，`d_theta` 是 frozen protected suffix。新建的 public head
`g_psi` 只消费 `s_theta` 的内部输出；`psi` 与 `theta` 分离，训练 public head 时 `theta` 必须冻结。
`P` 是由 V1-M1 input adapter 产生的 canonical uint8 snapshot 和固定 preprocessing。

C2 固定使用双入口，不接受请求 payload 中的 route selector：

```text
handle_public(image)

begin_protected(image, commitment)
respond_protected(response)
abort_protected(abort)
```

`handle_public` 由可信部署配置显式启用并绑定 C2 public route；它不需要 protected credential，不调用
V1-C1 verifier，只允许执行 `P -> s_theta -> g_psi`。这是有意设计的未认证公共能力，不是 protected
认证失败后的弱回退。部署未启用 public entry 时必须直接 deny，且不执行任何业务 module。

protected entry 继续使用 V1-P2/A3-v2 credential/transcript。只有 V1-C1 evidence 精确接受且唯一协调器
提交 `PROTECTED` 后，hard dispatcher 才允许执行 `P -> s_theta -> d_theta`。parse failure、relation
reject、tamper、replay、expiry、abort、route confusion、profile/model/input binding mismatch 和空状态
都不能自动调用 public entry。

请求 payload 不得传入、修改或选择 `entry`、`mode`、`cut`、`head`、`profile`、`threshold`、
`decision`、`expert_id`、verifier implementation 或 fallback policy。entry 由被调用的可信 API 决定；
cut、head、profile、threshold、model 和 policy 只由构造/加载时的可信配置固定，并只在该边界校验一次。

组合对象只拥有一份 accepted R2 module tree。split composition 必须直接复用该 tree 的 `stem`、
`layer1..layer4`、`average_pool`、`flatten` 和 `classifier`，不得为 prefix/suffix 复制参数，也不得把同一
R2 module 再注册到产生重复 `state_dict` ownership 的多个容器。公共 artifact 只能引用 accepted R2
digest，不能包含 R2 state。

## 4. Route decision and execution state

C2 将授权/路由决定与执行结果建模为两个正交维度：

```text
RouteDecision = PUBLIC | PROTECTED | DENY
ExecutionState = NOT_STARTED | RUNNING | SUCCEEDED | FAILED
```

`DENY` 专指业务执行开始前提交的拒绝，必须满足 public head 和 protected suffix 均零调用。
`PROTECTED + FAILED` 是 `PROTECTED_EXECUTION_ERROR`：verifier 已接受、协调器已提交 `PROTECTED`，
但 trusted preprocessing、prefix、suffix、top-1 extraction 或内部结果封装在提交后失败。它不能被内部
审计重标为 pre-execution `DENY`，不能回滚为 `PUBLIC`，不能释放 logits/features。`PUBLIC + FAILED`
同样保留已提交 public route 与执行失败的区别。

外部为避免泄露内部故障细节，可以把所有未成功释放业务结果的终态统一映射为固定 C2 deny envelope；
该外部 envelope 不改变内部 `RouteDecision` 和 `ExecutionState`。稳定审计只记录非敏感状态码、阶段和
计数，不记录图像、credential、transcript、features 或 logits。

protected 成功路径必须满足以下事件偏序：

```text
verifier_accept
  < coordinator_commit(PROTECTED)
  < preprocess_start
  < prefix_start
  < suffix_start
  < internal_result_commit
  < response_release
```

任一 protected suffix 实际启动在语义上必须蕴含：V1-C1 neural evidence 为 exact bound accept；由既有
全 canonical input `V_nn == V_ref` 证明可推出 `V_ref(canonical(response)) = 1`；协调器已经提交
`PROTECTED`；transcript 的唯一 claim 尚未被其他请求消费；model/profile/input binding 匹配。运行时
不得为此再次调用 `V_ref`，也不得在每个请求重扫可信 R2 权重或 topology。

每个 transcript 最多 claim 一次、提交一次、执行一次并向唯一可信 adapter 交付一次内部结果；C2 最多
释放一次外部 protected response。并发重复 response 只能有一个线程进入 verifier 后的 protected commit/
execution，其他请求固定 deny 且不增加 verifier/prefix/suffix 调用。已开始的执行不重试，不回滚 transcript。

## 5. A3-v2 internal result compatibility

C2 需要 protected 路径返回 100 类 top-1 `class_id`，因此允许对 A3-v2 做小范围内部重构，但不改变
A3-v2 wire/transcript 或 C1 公共 API。推荐内部边界为：

```text
coordinator.commit_and_execute(...) -> InternalExecutionResult
```

`InternalExecutionResult` 只能由唯一协调器构造，并携带内部 route decision、execution state、稳定错误码
和至多一个仅供可信 adapter 消费的暂态 operation value。它不是 evidence、capability 或外部可构造输入。
operation value 不进入日志或 artifact，交付一次后即由 adapter 丢弃。

- C1 adapter 消费并丢弃内部 operation value，对外继续返回原有 version-4 status-only
  `{"version": 4, "status": "protected"}` 或 deny；C1 message、response schema、计数和 zero-call 测试
  必须保持不变。
- C2 adapter 只在 `RouteDecision.PROTECTED + ExecutionState.SUCCEEDED` 时从内部 100 类 logits 计算一次
  top-1 `class_id`，随后丢弃 logits，对外返回 version-5 protected envelope。
- execution failure、top-1 extraction failure 或 envelope construction failure 均不释放内部 value，
  不 fallback public，并记录对应 execution stage。

C2 使用自己的 version-5 外部 envelope；A3-v2 内部 message/challenge/response bytes 继续沿用既有
canonical encoding。完整 C2 schema 为：

```text
CHALLENGE = {
  "version": 5,
  "status": "challenge",
  "message": bytes,
  "challenge": bytes,
  "transcript_id": bytes,
}

PUBLIC = {"version": 5, "status": "public", "coarse_class_id": 0..19}
PROTECTED = {"version": 5, "status": "protected", "class_id": 0..99}
DENY = {"version": 5, "status": "deny"}
```

响应不得包含 logits、features、verifier evidence、credential、identity/scope claims、capability、内部
route、execution state、protected suffix output 或错误堆栈。C1 version-4 和 C2 version-5 envelope 不得
在同一入口互相接受。

## 6. Public capability, cut and training selection

C2 的第一项 public capability 固定为 CIFAR-100 的 20-class coarse superclass prediction。coarse-label
target 必须直接使用已验证官方 archive 每条 record 的原生 `coarse_labels` 规范 index `0..19`，并在
C2 data manifest 中记录 ordered coarse-label digest；不得从 fine-label prediction、R2 logits、test
metrics 或事后聚类派生映射。public 响应不返回 100-class logits、R2 classifier output、prefix feature
或任何可重标记为 protected 的 token。

C2 只注册以下完整 residual-stage boundary candidate，顺序同时定义“最浅”关系：
`layer2 < layer3 < layer4`。每个 prefix 从 frozen R2 的 `stem` 开始并包含至该 candidate 的完整
residual stage；对应 protected suffix 从下一 stage 开始并包含原 R2 的
`average_pool -> flatten -> classifier`。不得在 residual block 内切分、跨 shortcut 切分、增加 feature
projection、复制 backbone 参数或改变 BatchNorm state。

三个 candidate 使用同一种 public-head family：

```text
AdaptiveAvgPool2d((1, 1)) -> Flatten(start_dim=1) -> Linear(C_cut, 20)
```

其中 `C_layer2=128`、`C_layer3=256`、`C_layer4=512`，含 bias 参数量分别为 `2,580`、`5,140`、
`10,260`。请求方不能选择 cut、head、label mapping 或 threshold。

public utility acceptance threshold 预注册为 validation/test coarse top-1 `>=75.00%`。C2 head-only
training run 使用两个独立 seed：H1=`1729`、H2=`1730`；H1/H2 只是 C2 public-head run 名称，不是
既有 protected baseline R1/R2 artifact 的身份。

训练和选择顺序固定如下：

1. H1 分别训练三个 candidate head，并完整报告三个 validation 结果；每个 run 只保存 validation top-1
   严格提高时的最早 checkpoint。按 `layer2, layer3, layer4` 顺序选择第一个 validation top-1
   `>=75.00%` 的最浅 cut，不以最高分或 test 指标改选更深 cut。三个 candidate 均失败则停止并报告
   C2 public utility 未闭合。
2. H2 只对 H1 选中的 cut 从独立初始化训练一次。H1/H2 在该 cut 的 validation top-1 均须
   `>=75.00%`，且绝对差不得超过 `2.00` percentage points；否则不验收、不重试、不更换 seed、
   不回退其他 cut，也不降低 threshold。
3. 通过后仅按同一 cut 的 validation top-1 在 H1/H2 head 中选择 accepted head；平局固定选择 H1。
   test split 只对该 accepted head 评估一次，test coarse top-1 必须 `>=75.00%`，不得参与 cut、epoch、
   run 或 threshold 选择。H1/H2 差异只称为稳定性复验，不解释为统计证明。

每次 head training 固定 `50` epochs、train batch `128`、validation/test batch `256`、现有 V1-M1
train/validation indices、normalization、random crop/flip、loader worker 和 deterministic CUDA policy；
loss 为无 label smoothing/class weighting/mixup/cutmix 的 `CrossEntropyLoss`，optimizer 为只接收
`g_psi` 参数的 SGD（`lr=0.1`、`momentum=0.9`、`weight_decay=0.0005`、`nesterov=True`），scheduler 为
`CosineAnnealingLR(T_max=50, eta_min=0.0)`。共享 R2 prefix/suffix 在全部阶段保持 `eval`、float32、
`requires_grad=False`，不得重训、微调、改变 accepted state 或以 public 指标更换 R2。

正式 public artifact 只能保存 selected public-head state、cut identifier、训练/选择 metrics、ordered
coarse-label digest、accepted R2/data/profile digest 和环境 provenance；不得复制或写出 R2 state、
prefix feature、图像、credential、transcript 或 logits。首次服务器运行必须在本地 runner、parser、
artifact writer、停止条件和全部测试闭合并发布完整 commit 后另行进入 `SERVER_REQUIRED`。

## 7. Protected semantic preservation

对最终 cut，direct R2 与 split protected path 必须使用同一 canonical uint8 snapshot、同一 preprocessing、
同一已加载 R2 module 实例、`eval()`、float32、同一 device、同一 deterministic policy 和相同 batch
execution shape。所有 R2 参数保持 `requires_grad=False`，模型不含 dropout 或其他请求期随机 module。

冻结的 A4000 tuple 已支持确定性 bitwise equality，因此 C2 主验收要求 direct/split 100 类 logits
bitwise equal；不得观察结果后放宽容差。若未来设备无法满足该执行契约，必须创建新 environment/profile
并在运行前固定容差，本 C2 profile 不自动降级。

报告必须包含：

- bitwise logits equality；
- max absolute error；
- max relative error，其中逐元素分母固定为 `max(abs(direct), abs(split), 1e-12)`；
- top-1 prediction equality；
- direct/split ordered logits digest；
- direct/split ordered prediction digest；
- accepted R2 canonical state digest 和既有 baseline prediction digest。

digest 编码必须复用 C1 已验收规则：每个 batch-1 logits 先转换为 CPU contiguous float32，按 test index
顺序把其 C-order raw bytes 输入 SHA-256；每个 top-1 prediction 按同一顺序编码为 big-endian signed int64
后输入 SHA-256。不得改用 JSON float、文本、小数截断、平台默认 integer width 或事后选择的序列化。

protected fine-class test accuracy 必须等于同一 execution shape 的 direct R2；accuracy 只验证业务语义，
不能替代逐元素 logits equality 和 digest。

## 8. Hard dispatcher, counters and event evidence

C2 禁止 `gate * logits`、专家输出加权和、soft/learned routing、先执行完整 R2 再隐藏 logits，或任何
通过输出遮蔽冒充零调用的方案。hard dispatcher 必须在业务 module 启动前依据协调器已提交的本地
`RouteDecision` 执行真实控制流：

```text
PUBLIC    -> preprocess -> prefix -> public_head
PROTECTED -> preprocess -> prefix -> protected_suffix
DENY      -> no business module
```

forward counter 与阶段 latency 属于实验 instrumentation，必须放在 `experiments/` wrapper、forward hook
或测试 probe 中，不写入 `access/` 安全核心的每请求路径。counter 在对应 module forward 开始时增加，
因此 forward 内异常仍计为一次 started call。协调器只保存协议正确性所需的 claim/commit/execution/
release 状态和稳定非敏感审计事件。

验收计数矩阵固定为：

| Route/outcome | Verifier | Prefix | Public head | Protected suffix | External result |
| --- | ---: | ---: | ---: | ---: | --- |
| public success | 0 | 1 | 1 | 0 | PUBLIC |
| protected success | 1 | 1 | 0 | 1 | PROTECTED |
| canonical relation reject/tampered response | 1 | 0 | 0 | 0 | DENY |
| malformed response/replay/expiry/abort/pre-verifier route mismatch | 0 | 0 | 0 | 0 | DENY |
| post-commit preprocessing error | 1 | 0 | 0 | 0 | fixed deny envelope; internal execution error |
| post-commit prefix error | 1 | 1 | 0 | 0 | fixed deny envelope; internal execution error |
| post-commit suffix error | 1 | 1 | 0 | 1 | fixed deny envelope; internal execution error |
| post-commit protected result/extraction error | 1 | 1 | 0 | 1 | fixed deny envelope; internal execution error |

“tamper”必须按发生阶段拆分：不能把 parser/claim 前的非规范 tamper 误报为一次 verifier call，也不能把
已 claim 后进入 V1-C1 relation 的 canonical tamper 误报为零 verifier call。public-head forward 异常为
`PUBLIC + FAILED`，计数为 verifier `0`、prefix `1`、public head `1`、suffix `0`，且只返回固定 deny
envelope。

实验事件必须验证成功路径的偏序，并验证所有 pre-execution deny 没有 `preprocess_start`、`prefix_start`、
`public_head_start` 或 `suffix_start`。任何 failure event 只能出现在其实际 started stage 之后；不能为满足
zero-call 报告而把 post-commit error 重新分类。

## 9. Test, report and artifact obligations

C2 必须分别验收，不能以单一 accuracy 或随机测试替代：

1. **Neural relation soundness:** 复用既有 V1-C1 全 canonical input `V_nn == V_ref` 证明，并回归
   malformed/noncanonical transcript fail-closed；C2 不重新训练或修改 `V_phi`。
2. **Protected semantic preservation:** 按 section 7 验收 direct/split logits、prediction 和 digest。
3. **Public utility:** 按 section 6 的 validation-only 规则选择，test 只评估最终 head 一次。
4. **Hard execution isolation:** public/protected/deny 及 post-commit error 的计数与 section 8 完全一致。
5. **Fail-closed:** tamper、replay、expiry、abort、route confusion、profile/model/input mismatch、空/非法
   state 均不能执行 protected suffix，也不能 fallback public。
6. **State correctness:** 每个 transcript 最多 claim/commit/execute/result-delivery/response-release 各一次；
   并发重复提交只有一个可进入 protected execution。
7. **Artifact integrity:** public artifact 不复制 R2 state，且 cut/head/profile/threshold/decision 不能由请求
   修改。

10,000-image protected report 必须为每张 canonical test snapshot 生成新鲜 commitment/challenge/response
transcript；绑定 image A 的 transcript 不得用于 image B。报告必须把 credential generation、response
encoding、verification、coordinator、preprocessing、prefix、public head、suffix、top-1 extraction 和
response release 开销分开，记录相同同步方法下的 p50/p95 latency 与 throughput。

正式 accepted-state report 至少包含：模型和 artifact hash、accepted R2 provenance、三个 cut 的 H1
validation、选中 cut 的 H2 稳定性复验、最终选择依据、public coarse accuracy、protected fine accuracy、
direct/split logits 对照与 digest、完整计数矩阵、分段 latency/throughput、tamper/replay/expiry/abort、
并发重复提交、route confusion、post-commit execution error、public artifact R2-state absence、全部
version-5 schema，以及所有失败案例和未满足条件。

commit `8455ff3d9e2c6f34edd0bbd2425c6ed4d9d5e1fd` 的 training runner 把 decoded-data digest 误写到
manifest 的 `accepted_r2_state_sha256` 字段；该错误只影响 provenance 标签，不改变已保存 public-head
state、H1/H2 选择或 `85.17%` test 结果。已有正式 artifact 不覆盖、不删除、不重训。evaluator 仅在原
manifest 该字段精确等于同一 manifest 的 `data.decoded_sha256`，且原 manifest/report/state 其余完整绑定
全部通过时，追加一次 `metadata-correction.json`；修正记录必须绑定三个原文件的 SHA-256，并写明冻结
accepted R2 canonical state digest。未来 runner 直接写入正确 state digest，不需要该修正文件。正式报告
必须同时记录原 manifest/report/state 摘要和 correction 摘要，不能静默改写历史 provenance。

可选 capability-leakage 实验不属于 C2 闭合条件。只有论文进一步声称 public 用户不能获得 protected
fine-grained information 时才必须执行，并预注册 attacker auxiliary data。至少比较 coarse-prior-only、
image-only surrogate 与 image-plus-public-output surrogate；训练只使用 attacker auxiliary train split，
test 仅作一次 fine-label recovery 评估。该实验只能量化增量经验泄露，不能证明信息论保密。

## 10. Transformation and M2 boundary

C2/M2 首期只支持已冻结的 Python/PyTorch module graph 与环境。state reload、module refactor、quantization、
pruning、fine-tuning、export 或其他变换默认使验收失效，必须拒绝部署或作为新 profile 重新执行 relation、
semantic-preservation 和 routing-isolation 验收。

M2 在 C2 二专家契约上增加受保护专家集合 `E_1...E_k`。canonical claims 只提供 transcript 绑定的
identity/scope 等证据字段，唯一协调器依据可信本地 policy/registry 提交不可变 `RouteContext` 与
`allowed_mask`；验证器、请求方和普通 router 均不能直接铸造权限掩码。任务得分只用于已授权集合内的
选择。若授权集合为空、scope 不匹配或路由失败，结果为 `DENY`，不能改走 `E0` 或其他受保护专家。

M2 还必须单独验收 route soundness：任何 `j > 0` 的专家实际执行都蕴含 reference verifier 接受、
协调器已提交 `PROTECTED`、`j` 属于可信 policy 生成且不大于 credential scope 的 mask。full-scope
credential 与完整 MoE reference 对照；restricted-scope credential 只与同一授权集合上的 constrained-
router reference 对照，不能错误要求它等于未受约束的完整 MoE。

## 11. Implementation increments and resources

| Increment | Resource | Required outcome |
| --- | --- | --- |
| Revised C2 decision freeze | `LOCAL_OK` | 本文档、工作日志、研究设计、安全边界和 V1-M1 决策一致；治理检查通过 |
| A3-v2 internal result refactor | `LOCAL_OK` | C1 version-4/wire/transcript/zero-call regression 不变；内部 operation value 最多交付一次 |
| M1-C2 split/composition and hard dispatcher | `LOCAL_OK` | direct/split logits 等价；双入口、version-5 schema、正交状态、调用矩阵与事件顺序测试通过 |
| M1-C2 public-head runner/artifact preflight | `LOCAL_OK` | coarse-label parser/digest、训练/选择 runner、artifact writer、停止条件和全部测试闭合并发布完整 commit |
| M1-C2 public-head training | `SERVER_REQUIRED` | 在冻结 R2、数据和预注册 H1/H2 规则下训练/选择 public head；不得重训 R2 |
| M1-C2 accepted-state report | `SERVER_REQUIRED` | 10,000-image public/protected/deny/error 报告、分段 latency、digest 和 artifact manifest 验收 |
| M2 multi-expert constrained routing | `LOCAL_OK` before formal model evaluation | 固定 scope-to-mask policy、受约束 router、空集合/route confusion 与逐专家零调用测试 |

在 C1 accepted-R2 正式报告完成前可以本地实现 split/composition、public-head interface、hard dispatcher、
instrumentation hooks、A3 内部结果对象和状态机测试，但不能用本机随机 R2 宣称 C2 正式闭合。当前 C1
报告已经闭合，因此本地实现可开始；正式 public-head 训练、正式 artifact 或 GPU 性能测量仍必须重新通过
服务器成本门槛。源码、测试、输入/artifact/environment、唯一命令、预期输出和停止条件未闭合并发布完整
commit 前，`PROJECT_WORKLOG.md` 必须保持 `LOCAL_OK`，不得要求项目负责人启动或保持计费服务器。

研究报告必须把 C1 最小 reference、C2 二专家硬路由、M2 多专家受约束路由、public head quality、
V1 neural relation soundness、协议安全假设和不支持的白盒/transform 场景分开叙述。
