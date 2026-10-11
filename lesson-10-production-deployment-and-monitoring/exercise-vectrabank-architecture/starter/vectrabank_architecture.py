"""
vectrabank_architecture.py - EXERCISE STARTER (Student-Led)
==============================================================
Module 10 Exercise: Plan a Production Deployment Architecture for VectraBank

Create a deployment architecture plan for VectraBank's financial services
multi-agent system. Define runtime configuration, monitoring strategy,
cost estimates, and operational runbooks.

Same planning pattern as the demo (deployment_walkthrough.py),
with additions:
  1. FINANCIAL DOMAIN — VectraBank-specific agents and KBs
  2. OPERATIONAL RUNBOOK (NEW) — deploy, rollback, kill switch, latency
  3. COMPLIANCE REQUIREMENTS — VPC network mode, stricter thresholds
  4. COST OPTIMIZATION — model selection recommendations

Instructions:
  - Follow the demo pattern (deployment_walkthrough.py)
  - Look for TODO 1-8 below
  - Define configs as Python dicts; the provided helper handles deployment
  - Focus on WHAT to configure and WHY
"""

import json
import os
from pathlib import Path
import agentcore_cli
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


# ═══════════════════════════════════════════════════════
#  VECTRABANK RUNTIME CONFIGURATION
# ═══════════════════════════════════════════════════════

# TODO 1: Define the AgentCore Runtime configuration
# Hint: Same structure as demo, but:
#   - Use networkMode "PUBLIC" and serverProtocol "HTTP" in this lab
#   - Document private subnets and security groups for a production VPC
#   - Role and guardrail IDs are discovered by the provided helper
#   - Environment variables: 3 KB IDs, state table, audit table, region, log level
VECTRABANK_RUNTIME_CONFIG = {
    "agentRuntimeName": "vectrabank-financial-services",
    "networkConfiguration": {
        "networkMode": "PUBLIC",  # For lab; Production requires "VPC"
        "productionVpcPlan": {
            "subnets": ["subnet-private-1", "subnet-private-2", "subnet-private-3"],
            "securityGroups": ["sg-vectrabank-agent-runtime"]
        }
    },
    "protocolConfiguration": {
        "serverProtocol": "HTTP"
    },
    "environmentVariables": {
        "MARKET_DATA_KB_ID": "kb-market-data-001",
        "COMPLIANCE_KB_ID": "kb-compliance-002",
        "RISK_KB_ID": "kb-risk-003",
        "STATE_TABLE": "vectrabank-agent-state",
        "AUDIT_TABLE": "vectrabank-compliance-audit",
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "LOG_LEVEL": "INFO"
    }
}


# ═══════════════════════════════════════════════════════
#  AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════

# TODO 2: Define 4 agents for VectraBank
# Hint: QueryRouter (Nova Lite), MarketDataRetriever (Nova Lite),
#       ComplianceRetriever (Nova Lite), FinancialAdvisor (Claude Sonnet)
#   Each needs: name, model, temperature, role, tools, estimated_tokens, requests_per_day
VECTRABANK_AGENTS = [
    {
        "name": "QueryRouter",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.0,
        "role": "Analyzes incoming requests and routes them to the appropriate specialized agents.",
        "tools": ["route_to_market", "route_to_compliance", "route_to_advisor"],
        "estimated_tokens_per_request": 400,
        "requests_per_day": 10000
    },
    {
        "name": "MarketDataRetriever",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.1,
        "role": "Retrieves real-time and historical market data.",
        "tools": ["query_market_data_kb"],
        "estimated_tokens_per_request": 1500,
        "requests_per_day": 6000
    },
    {
        "name": "ComplianceRetriever",
        "model": "amazon.nova-lite-v1:0",
        "temperature": 0.0,
        "role": "Retrieves financial regulations, internal policies, and compliance rules.",
        "tools": ["query_compliance_kb"],
        "estimated_tokens_per_request": 2000,
        "requests_per_day": 4000
    },
    {
        "name": "FinancialAdvisor",
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "temperature": 0.2,
        "role": "Synthesizes market and compliance data to generate safe, regulated financial insights.",
        "tools": ["generate_financial_report", "check_risk_profile"],
        "estimated_tokens_per_request": 3500,
        "requests_per_day": 3000
    }
]


# ═══════════════════════════════════════════════════════
#  MONITORING STRATEGY
# ═══════════════════════════════════════════════════════

# TODO 3: Define the monitoring dashboard (6 widgets)
# Hint: Total Queries, Latency P50/P99, Error Rate (2% threshold),
#       Guardrail Blocks, RAG Quality (NEW), Kill Switch Status (NEW)

# TODO 4: Define 3 alarms with thresholds
# Hint: HighErrorRate (2%), HighLatencyP99 (8s), GuardrailViolationSpike (50 blocks/5min)

# TODO 5: Define X-Ray tracing config
# Hint: 10% sampling for financial audit, annotations for query_type, agent_name, etc.
VECTRABANK_MONITORING = {
    "dashboard_name": "vectrabank-financial-services",
    "widgets": [
        "Total Queries", 
        "Latency P50/P99", 
        "Error Rate", 
        "Guardrail Blocks", 
        "RAG Quality", 
        "Kill Switch Status"
    ],
    "alarms": [
        {"name": "HighErrorRate", "threshold": "2%"},
        {"name": "HighLatencyP99", "threshold": "8s"},
        {"name": "GuardrailViolationSpike", "threshold": "50 blocks/5min"}
    ],
    "xray_tracing": {
        "sampling": "10%",
        "annotations": ["query_type", "agent_name", "client_tier", "compliance_status"]
    },
}


# ═══════════════════════════════════════════════════════
#  COST ESTIMATION (provided — same as demo)
# ═══════════════════════════════════════════════════════

MODEL_PRICING = {
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 0.003, "output": 0.015},
}


def estimate_monthly_costs(agents: list, days: int = 30) -> dict:
    """Estimate monthly costs for VectraBank."""
    costs = {}
    total = 0

    for agent in agents:
        model = agent["model"]
        pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.005})
        tokens = agent["estimated_tokens_per_request"]
        daily_requests = agent["requests_per_day"]

        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        daily_cost = ((input_tokens / 1000) * pricing["input"] +
                      (output_tokens / 1000) * pricing["output"]) * daily_requests
        monthly_cost = daily_cost * days

        costs[agent["name"]] = {
            "model": model, "daily_requests": daily_requests,
            "monthly_cost": round(monthly_cost, 2),
        }
        total += monthly_cost

    # TODO 6: Add infrastructure costs
    # Hint: DynamoDB (2 tables), Knowledge Bases (3 KBs), CloudWatch + X-Ray, VPC
    # Add each as costs["name"] = {"monthly_cost": estimated_cost}
    
    costs["Infrastructure_DynamoDB"] = {"model": "State & Audit Tables", "monthly_cost": 37.50}
    total += 37.50
    
    costs["Infrastructure_KBs"] = {"model": "3 Vector Databases", "monthly_cost": 120.00}
    total += 120.00
    
    costs["Infrastructure_CloudWatch_XRay"] = {"model": "Metrics, Logs, 10% Tracing", "monthly_cost": 46.00}
    total += 46.00
    
    costs["Infrastructure_VPC"] = {"model": "NAT Gateway & Endpoints", "monthly_cost": 57.85}
    total += 57.85

    costs["TOTAL"] = {"monthly_cost": round(total, 2)}
    return costs


# ═══════════════════════════════════════════════════════
#  OPERATIONAL RUNBOOK (NEW — not in demo)
# ═══════════════════════════════════════════════════════

# TODO 7: Define 4 operational runbook procedures
# Hint: Each is a dict with "name", "steps" (list of strings), and procedure-specific fields
#   1. Deploy a New Version — test, update guardrail, deploy runtime, smoke test, monitor
#   2. Rollback — identify previous version, update runtime, verify, post-mortem
#   3. Kill Switch Triggered — acknowledge, check audit log, investigate, fix, re-enable
#   4. Latency Investigation — X-Ray map, per-agent latency, identify bottleneck, scale
OPERATIONAL_RUNBOOK = {
    "deploy": {
        "name": "Production Deployment",
        "steps": [
            # TODO 7a: Add step 1 (run test suite)
            "1. Run full unit, integration, and compliance test suite.",
            # TODO 7b: Add step 2 (validate guardrail configuration)
            "2. Validate Bedrock Guardrail configuration.",
            # TODO 7c: Add step 3 (deploy new runtime version)
            "3. Deploy new runtime version via AgentCore CLI.",
            # TODO 7d: Add step 4 (run smoke tests on 10 test transactions)
            "4. Run smoke tests on 10 test transactions.",
            # TODO 7e: Add step 5 (monitor error rate and latency for 5 minutes)
            "5. Monitor error rate and latency for 5 minutes."
        ],
        "rollback_trigger": "Error rate > 2% or Guardrail evaluation failure",  # TODO 7f: Under what condition do we rollback? (e.g., "Error rate > 2%")
        "estimated_duration_minutes": 15,  # Estimate time to complete
    },

    "rollback": {
        "name": "Emergency Rollback",
        "steps": [
            # TODO 7g: Add step 1 (identify previous stable runtime version)
            "1. Identify previous stable runtime version ARN.",
            # TODO 7h: Add step 2 (revert runtime to previous version)
            "2. Revert runtime to previous version via AgentCore CLI update.",
            # TODO 7i: Add step 3 (verify agents are responding to transactions)
            "3. Verify agents are responding correctly to live transactions.",
            # TODO 7j: Add step 4 (post-incident review - what went wrong?)
            "4. Conduct post-incident review - what went wrong?"
        ],
        "estimated_duration_minutes": 10,
    },

    "kill_switch": {
        "name": "Kill Switch Activation",
        "steps": [
            # TODO 7k: Add step 1 (acknowledge alert in PagerDuty)
            "1. Acknowledge alert in PagerDuty.",
            # TODO 7l: Add step 2 (check AWS X-Ray for error patterns)
            "2. Check AWS X-Ray for error patterns.",
            # TODO 7m: Add step 3 (check CloudWatch audit logs for anomalies)
            "3. Check CloudWatch audit logs for adversarial anomalies.",
            # TODO 7n: Add step 4 (disable runtime if attack/fraud detected)
            "4. Disable runtime if attack/fraud detected.",
            # TODO 7o: Add step 5 (notify compliance team)
            "5. Notify compliance and security teams."
        ],
        "threshold": "Guardrail blocks > 50/5min",  # TODO 7p: What metric threshold triggers this? (e.g., "Guardrail blocks > 50/5min")
        "requires_approval": True,  # Financial system requires human approval
        "estimated_duration_minutes": 5,
    },

    "latency_investigation": {
        "name": "High Latency Investigation",
        "steps": [
            # TODO 7q: Add step 1 (pull X-Ray service map to identify slow service)
            "1. Pull X-Ray service map to identify slow service.",
            # TODO 7r: Add step 2 (check per-agent latency in CloudWatch)
            "2. Check per-agent latency in CloudWatch.",
            # TODO 7s: Add step 3 (identify bottleneck - agent, KB, or knowledge base retrieval?)
            "3. Identify bottleneck - agent, KB, or knowledge base retrieval?",
            # TODO 7t: Add step 4 (check DynamoDB throttling or KB latency)
            "4. Check DynamoDB throttling or KB backend latency.",
            # TODO 7u: Add step 5 (scale the bottleneck component - more provisioned capacity?)
            "5. Scale the bottleneck component - more provisioned capacity."
        ],
        "trigger_threshold": "P99 latency > 8 seconds",
        "estimated_duration_minutes": 20,
    },
}


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def deploy_to_agentcore() -> str:
    """Provided infrastructure; student architecture TODOs remain above."""
    required = ("networkConfiguration", "protocolConfiguration", "environmentVariables")
    if any(key not in VECTRABANK_RUNTIME_CONFIG for key in required) or len(VECTRABANK_AGENTS) != 4:
        raise ValueError("Complete the runtime configuration and four agent definitions before deployment.")
    resources = agentcore_cli.load_resources("lesson-10-exercise")
    config = {**VECTRABANK_RUNTIME_CONFIG,
              "roleArn": resources["AgentCoreRoleArn"],
              "guardrailConfiguration": {
                  "guardrailIdentifier": resources["GuardrailId"],
                  "guardrailVersion": os.environ.get("GUARDRAIL_VERSION", "DRAFT")}}
    return agentcore_cli.deploy(config)


def main():
    print("=" * 70)
    print("  VectraBank Deployment Architecture — Module 10 Exercise")
    print("  Runtime Config + Monitoring + Cost + Operational Runbook")
    print("=" * 70)

    # ── Runtime Configuration ──
    print(f"\n{'━' * 70}")
    print("  1. AgentCore Runtime Configuration")
    print(f"{'━' * 70}")

    # TODO 8: Print the runtime config, agent definitions, monitoring strategy,
    #         cost estimates, and operational runbook
    # Hint: Follow the demo's main() output format
    print(json.dumps(VECTRABANK_RUNTIME_CONFIG, indent=2))

    print(f"\n{'━' * 70}")
    print("  2. Agent Definitions & Model Assignments")
    print(f"{'━' * 70}")
    for agent in VECTRABANK_AGENTS:
        print(f"  - {agent['name']}: {agent['model']} (Role: {agent['role']})")

    print(f"\n{'━' * 70}")
    print("  3. Monitoring Strategy & Runbooks")
    print(f"{'━' * 70}")
    print("  Widgets:", ", ".join(VECTRABANK_MONITORING["widgets"]))
    print("  Alarms:")
    for alarm in VECTRABANK_MONITORING["alarms"]:
        print(f"    - {alarm['name']}: threshold {alarm['threshold']}")
    print(f"  X-Ray Sampling: {VECTRABANK_MONITORING['xray_tracing']['sampling']}")

    print(f"\n{'━' * 70}")
    print("  4. Operational Runbook")
    print(f"{'━' * 70}")
    for key, runbook in OPERATIONAL_RUNBOOK.items():
        print(f"\n  [{runbook['name'].upper()}]")
        for step in runbook['steps']:
            print(f"    {step}")

    if VECTRABANK_AGENTS:
        # ── Cost Estimation ──
        print(f"\n{'━' * 70}")
        print("  5. Monthly Cost Estimation (10,000 requests/day)")
        print(f"{'━' * 70}")
        costs = estimate_monthly_costs(VECTRABANK_AGENTS)
        print(f"\n  {'Component':<25s} {'Model':<40s} {'Monthly':>10s}")
        print(f"  {'─' * 75}")
        for name, data in costs.items():
            if name == "TOTAL":
                print(f"  {'─' * 75}")
            model = data.get("model", "—")
            cost = data["monthly_cost"]
            print(f"  {name:<25s} {model:<40s} ${cost:>9.2f}")

    print(f"\n  Key Takeaways:")
    print(f"  1. VPC NETWORK MODE — financial services agents stay internal")
    print(f"  2. MULTI-MODEL COST OPTIMIZATION — Lite for routing/retrieval, Sonnet for synthesis")
    print(f"  3. STRICTER THRESHOLDS — 2% error rate for financial compliance")
    print(f"  4. OPERATIONAL RUNBOOK — deploy, rollback, kill switch, latency procedures (NEW)")
    print(f"  5. AUDIT TRAIL — X-Ray at 10% sampling + full guardrail audit log\n")

    deploy_to_agentcore()


if __name__ == "__main__":
    main()