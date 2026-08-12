# Differential tests

本目录保存 `V_ref` 与 A1-C1 `CAN-RELU-EXACT-v1` conformance backend 的逐输入差分测试。测试穷尽每个分量的 canonical coefficient，复用 A0 边界向量，并分别检查 false accept 与 issuer-core false reject；这些有限 toy 域结果不证明不可伪造性或生产安全。
