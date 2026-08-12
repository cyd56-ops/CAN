#!/usr/bin/env bash

set -euo pipefail

readonly agent_file="AGENTS.md"
readonly readme_file="README.md"
readonly worklog_file="PROJECT_WORKLOG.md"
readonly security_file="SECURITY.md"
readonly research_file="docs/RESEARCH_DESIGN.md"
readonly a0_spec_file="docs/A0_PROTOCOL_SPEC.md"
readonly a1_spec_file="docs/A1_NUMERICAL_SPEC.md"
readonly a1_construction_file="docs/A1_CONSTRUCTION_DECISION.md"
readonly a1_backend_file="docs/A1_BACKEND_DECISION.md"
readonly a2_protocol_file="docs/A2_MODEL_EXPERIMENT_PROTOCOL.md"
readonly a2_capability_file="docs/A2_CAPABILITY_EXPERIMENT_SPEC.md"
readonly a3_protocol_file="docs/A3_CHALLENGE_RESPONSE_SPEC.md"
readonly a4_relation_file="docs/A4_GPV_RELATION_SPEC.md"
readonly a4_neural_file="docs/A4_NEURAL_CONSTRUCTION_DECISION.md"
readonly v1_protocol_file="docs/V1_PROTOCOL_SELECTION_DECISION.md"
readonly v1_module_protocol_file="docs/V1_MODULE_SIS_PROTOCOL_DECISION.md"
readonly v1_prover_sampler_file="docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md"
readonly v1_model_experiment_file="docs/V1_MODEL_EXPERIMENT_DECISION.md"

bootstrap_files=(
    ".gitignore"
    "README.md"
    "pyproject.toml"
    "requirements-dev.lock"
    "requirements-ml.lock"
    "tests/conftest.py"
    "src/can/__init__.py"
    "src/can/access/__init__.py"
    "src/can/access/a2_capability.py"
    "src/can/access/a2_gate.py"
    "src/can/access/a3_protocol.py"
    "src/can/access/a3_v2.py"
    "src/can/access/a4_adapter.py"
    "src/can/access/v1_adapter.py"
    "src/can/experiments/__init__.py"
    "src/can/experiments/a2_baseline.py"
    "src/can/experiments/a2_gate.py"
    "src/can/experiments/a2_public_baseline.py"
    "src/can/experiments/a2_capability.py"
    "src/can/experiments/v1_psr.py"
    "src/can/model/__init__.py"
    "src/can/model/a2_mlp.py"
    "src/can/model/a2_public_mlp.py"
    "src/can/reference/__init__.py"
    "src/can/reference/a0.py"
    "src/can/reference/a4.py"
    "src/can/reference/v1.py"
    "src/can/verifier/__init__.py"
    "src/can/verifier/a1.py"
    "src/can/verifier/a4.py"
    "src/can/verifier/v1.py"
    "tests/differential/README.md"
    "tests/differential/test_a1_differential.py"
    "tests/differential/test_a4_neural_differential.py"
    "tests/differential/test_v1_reference_differential.py"
    "tests/differential/test_v1_psr_differential.py"
    "tests/differential/test_v1_neural_differential.py"
    "tests/_v1_support.py"
    "tests/unit/test_a0_reference.py"
    "tests/unit/test_a1_verifier.py"
    "tests/unit/test_a2_baseline.py"
    "tests/unit/test_a2_capability.py"
    "tests/unit/test_a2_capability_experiment.py"
    "tests/unit/test_a2_gate.py"
    "tests/unit/test_a2_gate_experiment.py"
    "tests/unit/test_a2_mlp.py"
    "tests/unit/test_a2_public_baseline.py"
    "tests/unit/test_a2_public_mlp.py"
    "tests/unit/test_a3_protocol.py"
    "tests/unit/test_a4_reference.py"
    "tests/unit/test_a4_neural.py"
    "tests/unit/test_a3_v2.py"
    "tests/unit/test_v1_reference.py"
    "tests/unit/test_v1_psr.py"
    "tests/unit/test_v1_neural.py"
    "tests/unit/test_package.py"
    "tests/integration/test_a2_gate_integration.py"
    "tests/integration/test_a2_capability_integration.py"
    "tests/integration/test_package_boundaries.py"
    "tests/integration/test_a4_a3_integration.py"
    "tests/integration/test_v1_a3_v2_integration.py"
    "tests/integration/test_v1_neural_a3_v2_integration.py"
    "tests/security/test_a0_reference_security.py"
    "tests/security/test_a1_verifier_security.py"
    "tests/security/test_a2_baseline_security.py"
    "tests/security/test_a2_capability_security.py"
    "tests/security/test_a2_capability_experiment_security.py"
    "tests/security/test_a2_gate_experiment_security.py"
    "tests/security/test_a2_gate_security.py"
    "tests/security/test_a2_public_baseline_security.py"
    "tests/security/test_a3_protocol_security.py"
    "tests/security/test_a4_reference_security.py"
    "tests/security/test_a4_neural_security.py"
    "tests/security/test_v1_a3_v2_security.py"
    "tests/security/test_v1_psr_security.py"
    "tests/security/test_v1_neural_security.py"
    "tests/security/test_artifact_ignores.py"
)

for file in \
    "$agent_file" \
    "$worklog_file" \
    "$security_file" \
    "$research_file" \
    "$a0_spec_file" \
    "$a1_spec_file" \
    "$a1_construction_file" \
    "$a1_backend_file" \
    "$a2_protocol_file" \
    "$a2_capability_file" \
    "$a3_protocol_file" \
    "$a4_relation_file" \
    "$a4_neural_file" \
    "$v1_protocol_file" \
    "$v1_module_protocol_file" \
    "$v1_prover_sampler_file" \
    "$v1_model_experiment_file"; do
    if [[ ! -s "$file" ]]; then
        echo "missing or empty required file: $file" >&2
        exit 1
    fi
done

for file in "${bootstrap_files[@]}"; do
    if [[ ! -s "$file" ]]; then
        echo "missing or empty bootstrap file: $file" >&2
        exit 1
    fi
done

agent_headings=(
    "# Project"
    "# Session workflow"
    "# Engineering rules"
    "# Architecture and authorization rules"
    "# Cryptography rules"
    "# Testing rules"
    "# Git and worktree safety"
    "# Definition of done"
)

worklog_headings=(
    "# 1. Project goal and non-goals"
    "# 2. Architecture and security invariants"
    "# 3. Current state"
    "# 4. Milestones"
    "# 5. Task board"
    "# 6. Current next step"
    "# 7. Blockers and residual risks"
    "# 8. Recent work log"
    "# 9. Decisions"
)

security_headings=(
    "# Security"
    "## Status and scope"
    "## Trust model"
    "## Protected assets"
    "## Attacker capabilities"
    "## Trust boundaries"
    "## Input validation rules"
    "## Authentication and authorization flow"
    "## Key lifecycle"
    "## Replay and tamper protection"
    "## Fail-closed behavior"
    "## Audit and observability"
    "## Required security tests"
    "## Explicitly unsupported guarantees"
    "## Toy and experimental components"
)

research_headings=(
    "# Lattice-Based Neural Model Access Control: Research Design"
    "## 1. Document status"
    "## 2. Core research question"
    "## 3. Claims taxonomy"
    "## 4. Research hypotheses"
    "## 5. Prior work and novelty boundary"
    "## 6. Stage A architecture"
    "## 7. Stage A research increments"
    "## 8. Formal correctness and security obligations"
    "## 9. Stage B architecture"
    "## 10. Experiment and test matrix"
    "## 11. Planned implementation stack"
    "## 12. Publication strategy"
    "## 13. Explicit non-goals"
    "## 14. Open decisions"
)

a0_spec_headings=(
    "# A0 Toy LWE Numerical Unlock Protocol"
    "## 1. Status and claim boundary"
    "## 2. Roles and trust assumptions"
    "## 3. Fixed toy profile"
    "## 4. Runtime parameter generation"
    "## 5. Token issuance relation"
    "## 6. Canonical wire encoding"
    "## 7. Exact modular semantics"
    "## 8. Future neural error contract"
    "## 9. Reference oracle pseudocode"
    "## 10. Required test-vector families"
    "## 11. Attack analysis and non-guarantees"
    "## 12. Coordinator behavior"
    "## 13. Artifact and logging policy"
    "## 14. Acceptance criteria"
    "## 15. Deferred decisions"
)

a1_spec_headings=(
    "# A1 Numerical and Operator Contract"
    "## 1. Status and claim boundary"
    "## 2. Normative terms and trust boundary"
    "## 3. Canonical input and separation contract"
    "## 4. Trusted profile compilation"
    "## 5. Semantic tensors and exact ranges"
    "## 6. Exact operator semantics"
    "## 7. Rounding, saturation and overflow"
    "## 8. Error model and propagation budget"
    "## 9. Acceptance regions and proof obligations"
    "## 10. Evidence and authorization boundary"
    "## 11. Required conformance and differential tests"
    "## 12. Freeze, training and artifact rules"
    "## 13. Deferred construction hypotheses"
    "## 14. Acceptance criteria for this specification"
)

a1_construction_headings=(
    "# A1 Fixed ReLU Construction Decision"
    "## 1. Status and scope"
    "## 2. Decision summary"
    "## 3. Allowed operator and claim boundary"
    "## 4. Trusted compilation choice"
    "## 5. Primary network construction"
    "## 6. Formal correctness argument"
    "## 7. Numeric profile and range ledger"
    "## 8. Depth, width and operation accounting"
    "## 9. Comparison baseline and no-fallback rule"
    "## 10. Candidate disposition"
    "## 11. Implemented conformance backend contract"
    "## 12. Required proof and test artifacts"
    "## 13. Security implications and residual risks"
    "## 14. Acceptance criteria for this decision"
)

a1_backend_headings=(
    "# A1 PyTorch Exact-Integer Backend Decision"
    "## 1. Status and claim boundary"
    "## 2. Decision summary"
    "## 3. Read-only environment probe"
    "## 4. Version and installation channel"
    "## 5. Runtime and device contract"
    "## 6. Tensor and module layout"
    "## 7. Affine and ReLU operator mapping"
    "## 8. Overflow, rounding and saturation"
    "## 9. Quantization disposition"
    "## 10. Activation gate and fail-closed behavior"
    "## 11. Required conformance and differential tests"
    "## 12. Compiled artifact policy"
    "## 13. Performance and portability claims"
    "## 14. Disable, rollback and migration conditions"
    "## 15. Next implementation checkpoint boundary"
    "## 16. Official references"
    "## 17. Acceptance criteria for this decision"
)

a2_protocol_headings=(
    "# A2 Minimum Business Model Experiment Protocol"
    "## 1. Status and claim boundary"
    "## 2. Decision summary"
    "## 3. Framework and installation contract"
    "## 4. Dataset source, identity and license"
    "## 5. Canonical preprocessing and split"
    "## 6. Business model contract"
    "## 7. Deterministic training protocol"
    "## 8. Baseline metrics and acceptance"
    "## 9. Evidence, coordinator and model boundary"
    "## 10. Fail-closed response and zero-call contract"
    "## 11. Required tests"
    "## 12. Latency and overhead method"
    "## 13. Artifact, cache and cleanup policy"
    "## 14. Deferred and excluded scope"
    "## 15. Next implementation checkpoint boundary"
    "## 16. Official references"
    "## 17. Acceptance criteria for this decision"
)

a2_capability_headings=(
    "# A2-E2 Public/Protected Capability Experiment Specification"
    "## 1. Status and claim boundary"
    "## 2. Decision summary"
    "## 3. Trusted configuration and entry binding"
    "## 4. Public functional and output scope"
    "## 5. Protected functional and output scope"
    "## 6. Single-coordinator three-state semantics"
    "## 7. Non-upgrade and non-reuse rules"
    "## 8. Response envelopes"
    "## 9. Model and artifact separation"
    "## 10. Deterministic public baseline protocol"
    "## 11. Required acceptance matrix"
    "## 12. Experiment comparison and metrics"
    "## 13. Audit and observability"
    "## 14. Deferred and excluded scope"
    "## 15. Implementation checkpoint sequence"
    "## 16. Security and research interpretation"
    "## 17. Acceptance criteria for this specification"
)

a3_protocol_headings=(
    "# A3 Challenge-Response and Request-Binding Protocol"
    "## 1. Status and claim boundary"
    "## 2. Decision summary"
    "## 3. Roles and trust assumptions"
    "## 4. Fixed A3-v1 profile and typed request fields"
    "## 5. Canonical business-input digest"
    "## 6. Canonical 133-byte message encoding"
    "## 7. Challenge issuance protocol"
    "## 8. Proof-response parsing and binding checks"
    "## 9. Evidence-only verifier contract"
    "## 10. Trusted nonce lifecycle and state contract"
    "## 11. Atomic consume and coordinator commit order"
    "## 12. External responses, audit and information release"
    "## 13. Security games and formal properties"
    "## 14. Required acceptance matrix"
    "## 15. State, artifact and logging policy"
    "## 16. Deferred and excluded scope"
    "## 17. Implementation checkpoint sequence"
    "## 18. Acceptance criteria for this specification"
)

a4_relation_headings=(
    "# A4 GPV Public-Verification Relation Specification"
    "## 1. Status and claim boundary"
    "## 2. Relation selection and reviewed sources"
    "## 3. Roles and trust assumptions"
    "## 4. Fixed toy conformance profile"
    "## 5. Canonical public profile"
    "## 6. Canonical message and hash-to-syndrome"
    "## 7. Exact 105-byte proof encoding"
    "## 8. Exact public-verification relation"
    "## 9. Evidence-only A3 adapter contract"
    "## 10. Neural verifier reference contract"
    "## 11. Security analysis and non-guarantees"
    "## 12. Required test families"
    "## 13. Artifact and logging policy"
    "## 14. Implementation checkpoint sequence"
    "## 15. Acceptance criteria"
)

a4_neural_headings=(
    "# A4 Fixed ReLU Construction Decision"
    "## 1. Status and claim boundary"
    "## 2. Fixed canonical domain"
    "## 3. Exact reference predicate"
    "## 4. Construction decision"
    "## 5. Layer 1: norm violations and residual hinges"
    "## 6. Layer 2: exact point pulses and norm accumulator"
    "## 7. Layer 3: final fail-closed conjunction"
    "## 8. Multiple coverage and range ledger"
    "## 9. Topology and parameter accounting"
    "## 10. Completeness proof"
    "## 11. Soundness-preservation proof"
    "## 12. Allowed preprocessing and evidence boundary"
    "## 13. Required construction tests"
    "## 14. Deferred alternatives"
    "## 15. Acceptance criteria"
)

v1_protocol_headings=(
    "# V1 Lattice Identification Protocol Selection Decision"
    "## 1. Status and claim boundary"
    "## 2. Selected protocol and primary sources"
    "## 3. Candidate comparison and disposition"
    "## 4. Roles and trust assumptions"
    "## 5. Parameterized public profile"
    "## 6. Key relation"
    "## 7. A3-bound commit-first transcript"
    "## 8. Binding and transcript identifiers"
    "## 9. Canonical wire encodings"
    "## 10. Rejection sampling and abort semantics"
    "## 11. Exact verification relation"
    "## 12. A4-C1 compatibility decision"
    "## 13. Security games and proof separation"
    "## 14. Completeness and retry policy"
    "## 15. Evidence and authorization boundary"
    "## 16. Required tests for the next implementation checkpoint"
    "## 17. Deferred and excluded scope"
    "## 18. Acceptance criteria for this decision"
)

v1_module_protocol_headings=(
    "# V1 Module-SIS Sigma Protocol Selection Decision"
    "## 1. Status and claim boundary"
    "## 2. Selected protocol and primary sources"
    "## 3. Candidate disposition"
    "## 4. Roles and trust assumptions"
    "## 5. Parameterized module profile"
    "## 6. Key relation"
    "## 7. A3-v2 commit-first transcript"
    "## 8. Binding and transcript identifiers"
    "## 9. Canonical wire encodings"
    "## 10. Rejection and abort semantics"
    "## 11. Exact verification relation"
    "## 12. Neural construction boundary"
    "## 13. Security games and proof separation"
    "## 14. Completeness and retry policy"
    "## 15. Evidence and authorization boundary"
    "## 16. Required tests for the implementation checkpoints"
    "## 17. Deferred and excluded scope"
    "## 18. Acceptance criteria for this decision"
)

v1_prover_sampler_headings=(
    "# V1-P2 Non-Production Prover, Sampler and Rejection Experiment Specification"
    "## 1. Status and claim boundary"
    "## 2. Normative terms and trust boundary"
    "## 3. Fixed toy profile and generated-key separation"
    "## 4. Key relation and temporary secret lifecycle"
    "## 5. Canonical deterministic seed contract"
    "## 6. Exact sampler semantics"
    "## 7. Commitment and response computation"
    "## 8. Emit and abort rule"
    "## 9. Fresh-transcript retry policy"
    "## 10. Completeness and distribution obligations"
    "## 11. Test-vector plan"
    "## 12. Measurement contract"
    "## 13. Security and authorization boundary"
    "## 14. Theorem-condition separation"
    "## 15. Artifact and logging policy"
    "## 16. Deferred and excluded scope"
    "## 17. Implementation checkpoint sequence"
    "## 18. Acceptance criteria"
)

v1_model_experiment_headings=(
    "# V1 CIFAR-100 ResNet-18 Model Experiment Decision"
    "## 1. Status and claim boundary"
    "## 2. Decision summary"
    "## 3. Route and implementation isolation"
    "## 4. Dataset identity and supply-chain boundary"
    "## 5. Canonical V1 business input"
    "## 6. CIFAR-style ResNet-18 architecture"
    "## 7. Trusted preprocessing"
    "## 8. Training environment decision boundary"
    "## 9. Reproducibility protocol"
    "## 10. Baseline acceptance"
    "## 11. A3-v2 binding contract"
    "## 12. Coordinator and model boundary"
    "## 13. Gate experiment matrix"
    "## 14. Performance reporting"
    "## 15. Artifact and secret policy"
    "## 16. Required tests"
    "## 17. Deferred and excluded scope"
    "## 18. Acceptance criteria for this decision"
)

for heading in "${agent_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$agent_file"; then
        echo "missing AGENTS.md heading: $heading" >&2
        exit 1
    fi
done

for heading in "${worklog_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$worklog_file"; then
        echo "missing PROJECT_WORKLOG.md heading: $heading" >&2
        exit 1
    fi
done

for heading in "${security_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$security_file"; then
        echo "missing SECURITY.md heading: $heading" >&2
        exit 1
    fi
done

for heading in "${research_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$research_file"; then
        echo "missing research design heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a0_spec_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a0_spec_file"; then
        echo "missing A0 protocol heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a1_spec_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a1_spec_file"; then
        echo "missing A1 numerical spec heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a1_construction_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a1_construction_file"; then
        echo "missing A1 construction decision heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a1_backend_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a1_backend_file"; then
        echo "missing A1 backend decision heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a2_protocol_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a2_protocol_file"; then
        echo "missing A2 model experiment protocol heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a2_capability_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a2_capability_file"; then
        echo "missing A2 capability experiment heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a3_protocol_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a3_protocol_file"; then
        echo "missing A3 challenge-response protocol heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a4_relation_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a4_relation_file"; then
        echo "missing A4 GPV relation heading: $heading" >&2
        exit 1
    fi
done

for heading in "${a4_neural_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$a4_neural_file"; then
        echo "missing A4 neural construction heading: $heading" >&2
        exit 1
    fi
done

for heading in "${v1_protocol_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$v1_protocol_file"; then
        echo "missing V1 protocol selection heading: $heading" >&2
        exit 1
    fi
done

for heading in "${v1_module_protocol_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$v1_module_protocol_file"; then
        echo "missing V1 Module-SIS protocol heading: $heading" >&2
        exit 1
    fi
done

for heading in "${v1_prover_sampler_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$v1_prover_sampler_file"; then
        echo "missing V1 prover/sampler/rejection heading: $heading" >&2
        exit 1
    fi
done

for heading in "${v1_model_experiment_headings[@]}"; do
    if ! rg --fixed-strings --line-regexp --quiet "$heading" "$v1_model_experiment_file"; then
        echo "missing V1 model experiment heading: $heading" >&2
        exit 1
    fi
done

route_files=(
    "$readme_file"
    "$worklog_file"
    "$security_file"
    "$research_file"
    "$a3_protocol_file"
    "$a4_relation_file"
    "$a4_neural_file"
    "$v1_protocol_file"
    "$v1_module_protocol_file"
    "$v1_prover_sampler_file"
    "$v1_model_experiment_file"
)

for file in "${route_files[@]}"; do
    if ! rg --fixed-strings --quiet "V1-prep" "$file"; then
        echo "missing V1-prep route boundary: $file" >&2
        exit 1
    fi
done

next_step_count=$(rg --count '^\*\*唯一下一步：' "$worklog_file")
if [[ "$next_step_count" -ne 1 ]]; then
    echo "PROJECT_WORKLOG.md must contain exactly one global next step" >&2
    exit 1
fi

compute_status_count=$(rg --count '^\*\*计算资源：`(LOCAL_OK|SERVER_REQUIRED)`' "$worklog_file")
if [[ "$compute_status_count" -ne 1 ]]; then
    echo "PROJECT_WORKLOG.md must contain exactly one current compute-resource status" >&2
    exit 1
fi

if ! rg --fixed-strings --quiet 'SERVER_REQUIRED' "$agent_file" || \
   ! rg --fixed-strings --quiet 'SERVER_REQUIRED' "$v1_model_experiment_file"; then
    echo "missing persistent server-notification trigger" >&2
    exit 1
fi

while IFS= read -r status; do
    case "$status" in
        pending|in_progress|blocked|completed) ;;
        *)
            echo "invalid milestone or task status: $status" >&2
            exit 1
            ;;
    esac
done < <(
    awk -F '|' '
        /^\| M[0-9]+ / {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            print $4
        }
        /^# 5\. Task board$/ { in_tasks = 1; next }
        /^# 6\. Current next step$/ { in_tasks = 0 }
        in_tasks && /^\|/ && $4 !~ /^[[:space:]]*(Status|---)[[:space:]]*$/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            print $4
        }
    ' "$worklog_file"
)

echo "governance documentation check: PASS"
