#!/usr/bin/env python3
"""Uniform module access. Preparation is offline; execution needs a dispatcher.

An embedding runtime calls access(..., dispatcher=...) to execute its own native
interface. The CLI deliberately exposes read/prepare only: this construction
bundle does not launch robot experiments or guess a runtime configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

from common import ROOT, encode, payload, read_json


def access(request, data_root, protocol, dispatcher=None):
    module_id = request.get("module_id", "") if isinstance(request, dict) else ""
    if not isinstance(module_id, str):
        module_id = ""
    try:
        encode(request)  # Reject NaN/Infinity, which are not protocol JSON numbers.
        Draft202012Validator(protocol["request_schema"]).validate(request)
        cards = read_json(data_root / "01_cards.json")
        contracts = read_json(data_root / "02_contracts.json")["modules"]
        card, contract = cards[module_id], contracts[module_id]
        raw = payload(data_root, card)
        if request["operation"] == "read":
            if request.get("arguments") or "binding" in request:
                raise ValueError("read takes neither arguments nor a binding")
            data = {"content": raw.decode("utf-8"), "availability": card["source"]["availability"],
                    "sha256": card["source"]["sha256"]}
            status = "read"
        else:
            bindings = contract["access"]["bindings"]
            binding_name = request.get("binding")
            if binding_name is None:
                if len(bindings) != 1:
                    raise ValueError("select an explicit binding; no unambiguous native interface")
                binding_name = next(iter(bindings))
            binding = bindings[binding_name]
            arguments = request.get("arguments", {})
            Draft202012Validator(binding["arguments_schema"]).validate(arguments)
            native = {"protocol": binding["protocol"], "skill_id": binding["skill_id"],
                      "arguments": arguments, "binding": binding_name,
                      "metadata_module": binding["metadata_module"]}
            if "request_id" in request:
                native["request_id"] = request["request_id"]
            if request["operation"] == "prepare":
                data, status = native, "prepared"
            else:
                if dispatcher is None:
                    raise ValueError("invoke requires a runtime-supplied native dispatcher")
                if "request_id" not in request:
                    raise ValueError("invoke requires a stable request_id")
                # Exactly one delegation. Retry policy, permissions, observations,
                # budgets and backend version checks remain the runtime's job.
                data = dispatcher(native, binding)
                status = "executed"
        response = dict(ok=True, status=status, module_id=module_id, data=data, error=None)
    except (KeyError, ValueError, TypeError, OSError, UnicodeError) as exc:
        response = dict(ok=False, status="error", module_id=module_id, data=None,
                        error={"type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:
        # Schema and native-provider failures keep the same response envelope.
        response = dict(ok=False, status="error", module_id=module_id, data=None,
                        error={"type": type(exc).__name__, "message": str(exc)})
    Draft202012Validator(protocol["response_schema"]).validate(response)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--request", type=Path, help="JSON request file; otherwise read stdin")
    args = parser.parse_args()
    request = read_json(args.request) if args.request else json.load(sys.stdin)
    response = access(request, args.data, read_json(ROOT / "protocol.json"))
    print(json.dumps(response, ensure_ascii=False, indent=2))
    raise SystemExit(0 if response["ok"] else 1)
