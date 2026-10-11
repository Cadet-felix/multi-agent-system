"""
ecommerce_checkout_saga.py - EXERCISE STARTER (Student-Led)
==============================================================
Module 7 Exercise: Build a Saga with Compensations for E-Commerce Checkout

Architecture:
    Customer places checkout order
         │
    ┌────┴─────────────────────────────────────────────────┐
    │  Saga Orchestrator (Python, NOT LLM-driven)           │
    │  Forward: Inventory → Payment → Shipping (sequential) │
    │  Compensate: reverse order on failure                  │
    └────┬─────────────────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────────────────────────┐
    │  Saga State Machine (DynamoDB)                        │
    │  checkout_id (PK) | steps[] | overall_status | lock   │
    │  + Barrier counter: compensations_completed           │
    │  Saga resolves to 'failed' only when barrier reached  │
    └────┬─────────────────────────────────────────────────┘
         │
    Three checkout agents (each has forward + compensating action):
    ┌────┴─────────────────────────────────────────────────┐
    │ InventoryAgent:  reserve_items   / release_items      │
    │ PaymentAgent:    charge_card     / refund_card         │
    │ ShippingAgent:   schedule_delivery / cancel_delivery   │
    └──────────────────────────────────────────────────────┘

Same saga pattern as the demo (travel_booking_saga.py),
with one addition:
  BARRIER COORDINATION — an atomic counter that each compensation
  increments. Saga resolves to 'failed' only when the counter
  equals the number of steps to compensate.

Instructions:
  - Follow the demo pattern (travel_booking_saga.py)
  - Look for TODO 1-18 below
  - Saga state machine functions: create/update/lock/barrier
  - Each agent has a forward action and a compensating action
  - The orchestrator runs forward, detects failure, compensates in reverse
"""

import json
import re
import time
import logging
import os
import boto3
from decimal import Decimal
from datetime import datetime, timezone
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.models import BedrockModel

load_dotenv()

logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────
# HELPERS (provided)
# ─────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    """Strip <thinking>...</thinking> tags from Nova model outputs."""
    return re.sub(r"<thinking>.*?</thinking>\s*", "", str(text), flags=re.DOTALL).strip()


# NOTE: In production, extract shared helpers like run_agent_with_retry() and
# clean_response() to a common utils.py module to avoid code duplication.
def run_agent_with_retry(agent_builder, prompt: str, max_retries: int = 3) -> float:
    """Run an agent with retry logic for transient Bedrock errors.
    Uses exponential backoff (1s, 2s, 4s) to handle throttling."""
    for attempt in range(max_retries):
        try:
            agent = agent_builder()
            t = time.time()
            agent(prompt)
            return time.time() - t
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [Retry {attempt + 1}/{max_retries}] {e.__class__.__name__}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [Failed] {e.__class__.__name__} after {max_retries} attempts")
                raise


# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")

# ─────────────────────────────────────────────────────
# SAMPLE CHECKOUT DATA (provided)
# ─────────────────────────────────────────────────────
CHECKOUTS = [
    {
        "checkout_id": "CHK-001",
        "customer": "Alice Chen",
        "items": [
            {"sku": "LAPTOP-001", "name": "Pro Laptop 15\"", "qty": 1, "price": 1299.99},
            {"sku": "CASE-042", "name": "Laptop Sleeve", "qty": 1, "price": 49.99},
        ],
        "payment": {"method": "credit_card", "last4": "4242", "token": "tok_visa_success"},
        "shipping": {"address": "123 Main St, Seattle, WA 98101", "method": "express"},
        "simulate_failure": None,  # All succeed
    },
    {
        "checkout_id": "CHK-002",
        "customer": "Bob Martinez",
        "items": [
            {"sku": "PHONE-007", "name": "SmartPhone X", "qty": 2, "price": 899.99},
            {"sku": "CHARGER-01", "name": "Wireless Charger", "qty": 2, "price": 39.99},
        ],
        "payment": {"method": "credit_card", "last4": "0000", "token": "tok_insufficient_funds"},
        "shipping": {"address": "456 Oak Ave, Portland, OR 97201", "method": "standard"},
        "simulate_failure": "payment",  # Payment fails → release inventory
    },
    {
        "checkout_id": "CHK-003",
        "customer": "Carol Davis",
        "items": [
            {"sku": "DESK-100", "name": "Standing Desk", "qty": 1, "price": 599.99},
            {"sku": "MONITOR-22", "name": "4K Monitor 27\"", "qty": 1, "price": 449.99},
        ],
        "payment": {"method": "credit_card", "last4": "1234", "token": "tok_visa_success"},
        "shipping": {"address": "PO Box 999, Nowhere, XX 00000", "method": "express"},
        "simulate_failure": "shipping",  # Shipping fails → refund + release inventory
    },
]


# ═══════════════════════════════════════════════════════
#  DYNAMODB — Saga State Machine + Barrier (provided)
#  Real AWS DynamoDB resource (created by CloudFormation)
# ═══════════════════════════════════════════════════════

def to_dynamo(obj):
    """Convert Python objects to DynamoDB-compatible types (float→Decimal)."""
    return json.loads(json.dumps(obj), parse_float=Decimal)


def _dynamo_default(o):
    """Convert Decimal to int if whole number, float otherwise."""
    if isinstance(o, Decimal):
        return int(o) if o == int(o) else float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def from_dynamo(obj):
    """Convert DynamoDB types back to Python (Decimal→int or float)."""
    return json.loads(json.dumps(obj, default=_dynamo_default))


# DynamoDB checkout saga table (real AWS resource — created by CloudFormation)
CHECKOUT_SAGA_TABLE = os.environ.get("CHECKOUT_SAGA_TABLE", "lesson-07-saga-checkout-saga")
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
checkout_table = dynamodb.Table(CHECKOUT_SAGA_TABLE)


# ═══════════════════════════════════════════════════════
#  SAGA STATE MACHINE + BARRIER
#  Follow the same create/update/lock pattern from the demo,
#  then add increment_barrier() for the barrier (NEW).
# ═══════════════════════════════════════════════════════

# TODO 1: Implement create_saga(checkout_id, steps)
def create_saga(checkout_id: str, steps: list[str]) -> dict:
    record = {
        "checkout_id": checkout_id,
        "steps": [{"name": s, "status": "pending", "forward_ref": None, "compensation_ref": None} for s in steps],
        "current_phase": 0,
        "overall_status": "in_progress",
        "locked": False,
        "compensations_needed": 0,
        "compensations_completed": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ttl": int(time.time()) + 86400
    }
    checkout_table.put_item(Item=to_dynamo(record))
    return record


# TODO 2: Implement update_step(checkout_id, step_index, updates)
def update_step(checkout_id: str, step_index: int, updates: dict) -> dict:
    saga = get_saga(checkout_id)
    if not saga:
        return {}
    for key, val in updates.items():
        saga["steps"][step_index][key] = val
    checkout_table.put_item(Item=to_dynamo(saga))
    return saga


# TODO 3: Implement acquire_lock(checkout_id) -> bool
def acquire_lock(checkout_id: str) -> bool:
    try:
        checkout_table.update_item(
            Key={"checkout_id": checkout_id},
            UpdateExpression="SET locked = :true",
            ConditionExpression="locked = :false OR attribute_not_exists(locked)",
            ExpressionAttributeValues={":true": True, ":false": False}
        )
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return False
        raise


# TODO 4: Implement release_lock(checkout_id)
def release_lock(checkout_id: str):
    checkout_table.update_item(
        Key={"checkout_id": checkout_id},
        UpdateExpression="SET locked = :false",
        ExpressionAttributeValues={":false": False}
    )


# TODO 5: Implement increment_barrier(checkout_id) -> tuple[int, int]
def increment_barrier(checkout_id: str) -> tuple[int, int]:
    response = checkout_table.update_item(
        Key={"checkout_id": checkout_id},
        UpdateExpression="ADD compensations_completed :one",
        ExpressionAttributeValues={":one": 1},
        ReturnValues="ALL_NEW"
    )
    item = from_dynamo(response.get("Attributes", {}))
    return item.get("compensations_completed", 0), item.get("compensations_needed", 0)


def get_saga(checkout_id: str) -> dict | None:
    """Read current saga state (provided)."""
    response = checkout_table.get_item(Key={"checkout_id": checkout_id})
    item = response.get("Item")
    return from_dynamo(item) if item else None


# ═══════════════════════════════════════════════════════
#  CHECKOUT AGENTS — Each has forward + compensating action
#  Follow the same pattern as the demo's booking agents.
#  Each @tool function is provided — you implement the Agent builders.
# ═══════════════════════════════════════════════════════

# TODO 6-8: Build InventoryAgent (forward + cancel)
def build_inventory_agent(items: list, checkout_id: str,
                          cancel_mode: bool = False) -> Agent:
    """Agent for inventory reservation / release."""
    # TODO 6: Create BedrockModel with NOVA_LITE_MODEL, temperature=0.0
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)

    if cancel_mode:
        # TODO 7: Write system prompt for inventory release
        system_prompt = "You are an Inventory Agent in cancellation mode. Call the release_items tool to release reserved inventory. Report completion directly."

        @tool
        def release_items(checkout_id: str) -> str:
            """Release reserved inventory (compensating transaction).

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with release confirmation
            """
            comp_ref = f"REL-{checkout_id.split('-')[1]}"
            update_step(checkout_id, 0, {
                "status": "compensated",
                "compensation_ref": comp_ref,
            })

            # Increment barrier — this is the NEW pattern
            completed, needed = increment_barrier(checkout_id)
            print(f"      [Barrier] {completed}/{needed} compensations done")

            released = [{"sku": i["sku"], "qty": i["qty"]} for i in items]
            return json.dumps({
                "checkout_id": checkout_id,
                "action": "release_items",
                "released": released,
                "comp_ref": comp_ref,
            }, indent=2)

        return Agent(model=model, system_prompt=system_prompt, tools=[release_items])

    else:
        # TODO 8: Write system prompt for inventory reservation
        system_prompt = "You are an Inventory Agent. Call the reserve_items tool to reserve the required items. Report completion directly."

        @tool
        def reserve_items(checkout_id: str) -> str:
            """Reserve inventory for the checkout.

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with reservation confirmation
            """
            forward_ref = f"RSV-{checkout_id.split('-')[1]}-{int(time.time()) % 10000}"
            reserved = [{"sku": i["sku"], "name": i["name"], "qty": i["qty"]} for i in items]

            update_step(checkout_id, 0, {
                "status": "completed",
                "forward_ref": forward_ref,
            })

            return json.dumps({
                "checkout_id": checkout_id,
                "action": "reserve_items",
                "reserved": reserved,
                "forward_ref": forward_ref,
                "status": "confirmed",
            }, indent=2)

        # Build and return Agent
        return Agent(model=model, system_prompt=system_prompt, tools=[reserve_items])


# TODO 9-11: Build PaymentAgent (forward + cancel)
def build_payment_agent(payment_data: dict, total: float, checkout_id: str,
                        simulate_failure: bool = False,
                        cancel_mode: bool = False) -> Agent:
    """Agent for payment processing / refund."""
    # TODO 9: Create BedrockModel with NOVA_LITE_MODEL, temperature=0.0
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)

    if cancel_mode:
        # TODO 10: Write system prompt for refund
        system_prompt = "You are a Payment Agent in cancellation mode. Call the refund_card tool to reverse a previous charge. Report completion directly."

        @tool
        def refund_card(checkout_id: str) -> str:
            """Refund a previously charged card (compensating transaction).

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with refund confirmation
            """
            comp_ref = f"RFND-{checkout_id.split('-')[1]}"
            update_step(checkout_id, 1, {
                "status": "compensated",
                "compensation_ref": comp_ref,
            })

            completed, needed = increment_barrier(checkout_id)
            print(f"      [Barrier] {completed}/{needed} compensations done")

            return json.dumps({
                "checkout_id": checkout_id,
                "action": "refund_card",
                "amount": total,
                "last4": payment_data["last4"],
                "comp_ref": comp_ref,
            }, indent=2)

        return Agent(model=model, system_prompt=system_prompt, tools=[refund_card])

    else:
        # TODO 11: Write system prompt for charge
        system_prompt = "You are a Payment Agent. Call the charge_card tool to process payment. Report completion directly."

        @tool
        def charge_card(checkout_id: str) -> str:
            """Charge the customer's card.

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with payment result
            """
            if simulate_failure:
                update_step(checkout_id, 1, {"status": "failed"})
                return json.dumps({
                    "checkout_id": checkout_id,
                    "action": "charge_card",
                    "status": "failed",
                    "reason": "Insufficient funds",
                    "last4": payment_data["last4"],
                }, indent=2)

            forward_ref = f"PAY-{checkout_id.split('-')[1]}-{int(time.time()) % 10000}"
            update_step(checkout_id, 1, {
                "status": "completed",
                "forward_ref": forward_ref,
            })

            return json.dumps({
                "checkout_id": checkout_id,
                "action": "charge_card",
                "amount": total,
                "last4": payment_data["last4"],
                "forward_ref": forward_ref,
                "status": "confirmed",
            }, indent=2)

        return Agent(model=model, system_prompt=system_prompt, tools=[charge_card])


# TODO 12-14: Build ShippingAgent (forward + cancel)
def build_shipping_agent(shipping_data: dict, checkout_id: str,
                         simulate_failure: bool = False,
                         cancel_mode: bool = False) -> Agent:
    """Agent for shipping scheduling / cancellation."""
    # TODO 12: Create BedrockModel with NOVA_LITE_MODEL, temperature=0.0
    model = BedrockModel(model_id=NOVA_LITE_MODEL, region_name=AWS_REGION, temperature=0.0)

    if cancel_mode:
        # TODO 13: Write system prompt for delivery cancellation
        system_prompt = "You are a Shipping Agent in cancellation mode. Call the cancel_delivery tool to cancel a scheduled shipment. Report completion directly."

        @tool
        def cancel_delivery(checkout_id: str) -> str:
            """Cancel a scheduled delivery (compensating transaction).

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with cancellation confirmation
            """
            comp_ref = f"CXDEL-{checkout_id.split('-')[1]}"
            update_step(checkout_id, 2, {
                "status": "compensated",
                "compensation_ref": comp_ref,
            })

            completed, needed = increment_barrier(checkout_id)
            print(f"      [Barrier] {completed}/{needed} compensations done")

            return json.dumps({
                "checkout_id": checkout_id,
                "action": "cancel_delivery",
                "comp_ref": comp_ref,
            }, indent=2)

        return Agent(model=model, system_prompt=system_prompt, tools=[cancel_delivery])

    else:
        # TODO 14: Write system prompt for delivery scheduling
        system_prompt = "You are a Shipping Agent. Call the schedule_delivery tool to arrange the shipment. Report completion directly."

        @tool
        def schedule_delivery(checkout_id: str) -> str:
            """Schedule delivery for the order.

            Args:
                checkout_id: The checkout ID

            Returns:
                JSON with delivery scheduling result
            """
            if simulate_failure:
                update_step(checkout_id, 2, {"status": "failed"})
                return json.dumps({
                    "checkout_id": checkout_id,
                    "action": "schedule_delivery",
                    "status": "failed",
                    "reason": "Address undeliverable",
                    "address": shipping_data["address"],
                }, indent=2)

            forward_ref = f"SHIP-{checkout_id.split('-')[1]}-{int(time.time()) % 10000}"
            update_step(checkout_id, 2, {
                "status": "completed",
                "forward_ref": forward_ref,
            })

            return json.dumps({
                "checkout_id": checkout_id,
                "action": "schedule_delivery",
                "address": shipping_data["address"],
                "method": shipping_data["method"],
                "forward_ref": forward_ref,
                "status": "confirmed",
            }, indent=2)

        return Agent(model=model, system_prompt=system_prompt, tools=[schedule_delivery])


# ═══════════════════════════════════════════════════════
#  SAGA ORCHESTRATOR
#  TODO 15: Forward execution
#  TODO 16: Compensation with lock
#  TODO 17: Barrier check
#  TODO 18: Wire up main() to run all 3 scenarios
# ═══════════════════════════════════════════════════════

def run_checkout_saga(checkout: dict):
    """Execute a full saga for an e-commerce checkout."""
    checkout_id = checkout["checkout_id"]
    fail_at = checkout.get("simulate_failure")

    subtotal = sum(i["price"] * i["qty"] for i in checkout["items"])
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)

    print(f"\n  Creating saga state machine...")
    saga = create_saga(checkout_id, ["inventory", "payment", "shipping"])
    print(f"    State: {[s['name'] + '=' + s['status'] for s in saga['steps']]}")
    print(f"    Subtotal: ${subtotal:.2f}, Tax: ${tax:.2f}, Total: ${total:.2f}")

    # ── Forward Execution ────────────────────────────
    agents_config = [
        {
            "name": "inventory",
            "index": 0,
            "builder": lambda: build_inventory_agent(checkout["items"], checkout_id),
            "prompt": f"Reserve items for checkout {checkout_id}",
        },
        {
            "name": "payment",
            "index": 1,
            "builder": lambda: build_payment_agent(
                checkout["payment"], total, checkout_id,
                simulate_failure=(fail_at == "payment")
            ),
            "prompt": f"Charge card for checkout {checkout_id}",
        },
        {
            "name": "shipping",
            "index": 2,
            "builder": lambda: build_shipping_agent(
                checkout["shipping"], checkout_id,
                simulate_failure=(fail_at == "shipping")
            ),
            "prompt": f"Schedule delivery for checkout {checkout_id}",
        },
    ]

    # TODO 15: Forward execution — run agents sequentially, detect failure
    failed_step = None
    for config in agents_config:
        step_idx = config["index"]
        update_step(checkout_id, step_idx, {"status": "executing"})
        print(f"  [{config['name'].upper()}] Forward action running...")
        
        run_agent_with_retry(config["builder"], config["prompt"])
        
        saga = get_saga(checkout_id)
        if saga["steps"][step_idx]["status"] == "failed":
            failed_step = config
            print(f"    ✗ FAILED at {config['name']} step.")
            break

    saga = get_saga(checkout_id)
    if failed_step is None:
        checkout_table.update_item(
            Key={"checkout_id": checkout_id},
            UpdateExpression="SET overall_status = :status",
            ExpressionAttributeValues={":status": "completed"}
        )
        print(f"\n  ✓ Checkout {checkout_id} COMPLETED — order confirmed!")
        return get_saga(checkout_id)

    # TODO 16: Compensation with lock — compensate in REVERSE order
    compensation_builders = {
        "inventory": lambda: build_inventory_agent(checkout["items"], checkout_id, cancel_mode=True),
        "payment": lambda: build_payment_agent(checkout["payment"], total, checkout_id, cancel_mode=True),
        "shipping": lambda: build_shipping_agent(checkout["shipping"], checkout_id, cancel_mode=True),
    }
    compensation_prompts = {
        "inventory": f"Release items for checkout {checkout_id}",
        "payment": f"Refund card for checkout {checkout_id}",
        "shipping": f"Cancel delivery for checkout {checkout_id}",
    }
    
    print(f"\n  [SAGA TRIGGER] Failure detected. Initiating compensation saga...")
    
    checkout_table.update_item(
        Key={"checkout_id": checkout_id},
        UpdateExpression="SET overall_status = :status",
        ExpressionAttributeValues={":status": "compensating"}
    )
    
    if not acquire_lock(checkout_id):
        print("    [LOCK] Failed to acquire compensation lock (another process is rolling back).")
        return get_saga(checkout_id)
        
    print("    [LOCK] Compensation lock acquired.")
    
    steps_to_reverse = [s for s in saga["steps"] if s["status"] == "completed"]
    num_to_compensate = len(steps_to_reverse)
    
    checkout_table.update_item(
        Key={"checkout_id": checkout_id},
        UpdateExpression="SET compensations_needed = :needed",
        ExpressionAttributeValues={":needed": num_to_compensate}
    )
    
    for step in reversed(steps_to_reverse):
        name = step["name"]
        print(f"  [COMPENSATE] Reversing {name}...")
        builder = compensation_builders[name]
        prompt = compensation_prompts[name]
        run_agent_with_retry(builder, prompt)
        
    release_lock(checkout_id)
    print("    [LOCK] Compensation lock released.")

    # TODO 17: Barrier check — read saga, verify completed==needed, set status
    saga = get_saga(checkout_id)
    completed = saga.get("compensations_completed", 0)
    needed = saga.get("compensations_needed", 0)
    
    if completed == needed and needed > 0:
        checkout_table.update_item(
            Key={"checkout_id": checkout_id},
            UpdateExpression="SET overall_status = :status",
            ExpressionAttributeValues={":status": "failed"}
        )
        print(f"  ✓ SAGA RESOLVED: All {completed} compensations completed successfully. Checkout marked as 'failed'.")

    return get_saga(checkout_id)


# TODO 18: Wire up main() — loop through CHECKOUTS, run sagas, print results
def main():
    print("=" * 70)
    print("  E-Commerce Checkout Saga — Module 7 Exercise")
    print("  Saga Pattern with Compensating Transactions + Barrier")
    print("  3 Checkout Agents: Inventory → Payment → Shipping")
    print("=" * 70)
    
    for checkout in CHECKOUTS:
        print(f"\n{'━' * 70}")
        print(f"  Processing Checkout: {checkout['checkout_id']} for {checkout['customer']}")
        print(f"{'━' * 70}")
        
        final_saga = run_checkout_saga(checkout)
        
        print(f"\n  Final Saga State for {checkout['checkout_id']}:")
        print(f"    Overall Status: {final_saga['overall_status']}")
        for step in final_saga["steps"]:
            print(f"    - {step['name']}: {step['status']}")


if __name__ == "__main__":
    main()