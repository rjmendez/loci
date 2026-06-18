# Infrastructure and Development Rules
# Infrastructure and operations rules
# Generated: 2026-06-13

## Secrets Management

Never commit secrets — private keys, API tokens, database passwords, or any
credential — to version control. Store them in environment-specific secret
managers (Kubernetes Secrets, .env files excluded from git, vault services) and
restrict file permissions to 600 or tighter. Rotate any secret that was ever
committed before the exclusion rule was in place.
See skill: homelab-wireguard-vpn, homelab-pihole-dns, kubernetes-patterns.

## Change Management

Before executing any infrastructure change — schema migration, firewall rule,
VLAN move, or deployment — document a concrete rollback procedure and confirm it
is reachable from the target environment. Apply changes in stages, verify each
stage passes acceptance criteria, and only proceed when rollback is still
possible. Never alter production systems manually; every change must go through
a recorded, reproducible path. Ad-hoc manual edits to production are prohibited.
See skill: database-migrations, git-workflow, network-config-validation, homelab-network-readiness.

## Least Privilege

Grant the minimum permissions required for each component to perform its
function and no more. In Kubernetes, build a ServiceAccount → Role → RoleBinding
chain scoped to the namespace. In networks, isolate trust zones (IoT, guest,
server, management) with explicit firewall rules rather than broad ACCEPT
policies. Never expose management interfaces directly to the internet.
See skill: kubernetes-patterns, homelab-wireguard-vpn, homelab-vlan-segmentation.

## Version Pinning

Always specify exact version tags for container images, package dependencies,
and SDK references in production deployments. Use digest pinning (SHA256) for
the highest reproducibility guarantee. Treat version upgrades as deliberate,
reviewable changes — not silent automatic updates. Floating 'latest' tags are
acceptable only in throwaway local experimentation.
See skill: homelab-pihole-dns, deployment-patterns, mcp-server-patterns.

## Separation Of Concerns

Decompose systems into single-responsibility layers with clean boundaries:
separate schema migrations from data migrations; separate business logic from
data access; separate transport/protocol code from application logic. Apply the
same principle to container images — use multi-stage builds so build tools are
absent from the production layer. Mixing these boundaries complicates testing,
rollback, and auditing.
See skill: database-migrations, backend-patterns, mcp-server-patterns, deployment-patterns.

## Resource Limits

Every workload must declare explicit resource requests and limits. Every LLM or
external API pipeline must carry a budget tracker that halts execution when the
limit is reached, not just log a warning. In Kubernetes, absence of resource
limits is a deployment gate failure. For cost-sensitive pipelines, pre-check
budget before each call and surface the over-budget state as an error.
See skill: kubernetes-patterns, cost-aware-llm-pipeline, agent-payment-x402.

## Transient-Only Retry

Classify errors before retrying: transient errors (network timeouts, 429 rate
limits, 5xx server errors) may be retried with exponential backoff and a maximum
attempt count. Client errors (400, 401, 403, 422) must not be retried — they
indicate a bug in the caller. Authentication errors should surface immediately
and halt the pipeline.
See skill: cost-aware-llm-pipeline, api-design.

## Health Checks

Distinguish internal health (process alive) from external reachability (traffic
actually arriving). Kubernetes readiness probes must validate the full request
path, not just process liveness. After any deployment or network change, verify
connectivity from outside the service boundary. Canary and blue-green promotions
must gate on external health signals before switching traffic.
See skill: kubernetes-patterns, deployment-patterns, docker-patterns.

## Connection Pooling

All connections to shared stateful resources (PostgreSQL, Redis, external APIs)
must go through a connection pool with bounded size, idle timeout, and connection
validation. Never open and close a raw connection per request in production code.
Set TTLs on cached values to prevent stale data accumulation. Apply backpressure
patterns (rate limiting, queue depth limits) to prevent downstream overload.
See skill: redis-patterns, postgres-patterns, backend-patterns.

## Read-Only Start

Before issuing any write, destructive command, or configuration change: read and
inventory the current state, identify risks and affected components, and produce
a staged plan with validation criteria. This applies equally to autonomous agents
and to humans following runbooks. Document what you observed before documenting
what you changed.
See skill: ecc-tools-cost-audit, homelab-network-readiness, network-config-validation, database-migrations.
