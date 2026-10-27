#!/usr/bin/env python3
# Synthetic tool-call validity gate for the qwen3_xml parser path.
#
# Why this exists: the 2026-10-23 W8A8 certification (needles, accept-window,
# corpus floors) never measured tool-call validity, and production agent
# traffic later showed an abnormal error rate (results.md, Production incident
# 2026-10-26). Any change to verifier logits needs a gate on exactly that axis.
#
# Each case is a chat request with tools, generated deterministically from
# --salt (same salt means the same prompts across boots, the bench.py law), and
# scored on: the server emitted tool_calls at all, the function name is the
# expected one, the arguments parse as JSON, and every argument matches its
# declared check. A case with no tool_calls is what a client sees as "the model
# ignored my tools": the parser found no grammar and returned prose.
#
# Families probe distinct grammar stress: string and numeric and bool args,
# enums, arrays, nested objects, selecting one tool from eight by semantics,
# two parallel calls, the same call after a ~20k-token filler (prefill-sized
# batches, where the dense W8A8 arm engages), and continuing a conversation
# that already contains two completed tool round-trips.
#
# Usage on the remote box (greedy, serial, exits non-zero below --min-valid):
#   ./tool_call_battery.py --base-url http://fender.lan:8000 --salt 152001
#   ./tool_call_battery.py --only long_context --n 3 --json /tmp/tcb.json
#
# Keep --conc at 1 when the box serves traffic: this tool is a validity gate,
# not a load generator, and long_context cases are real prefills.

import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "stage", "route", "cargo", "anchor", "ledger", "pilot", "signal",
    "meadow", "engine", "valve", "harbor", "quartz", "atlas", "beacon",
    "drum", "ferry", "gadget", "hinge", "inlet", "jitter", "karma", "loom",
    "magnet", "notch", "onion", "prism", "quill", "relay", "splice", "taper",
    "ulna", "vortex", "wedge", "yonder", "zenith", "amber", "billet",
    "cipher", "digest", "echo", "flux", "glyph", "hoist", "ivory",
    "junction", "kiln", "lapis", "mosaic", "nectar", "oracle", "pellet",
    "quota", "rumble", "slab",
)


def filler(rng, words):
    """Deterministic pseudo-document, roughly words x 1.3 tokens."""
    out = []
    for _ in range(words // 12 + 1):
        n = rng.randint(8, 14)
        sent = [rng.choice(WORDS) for _ in range(n)]
        out.append(" ".join(sent).capitalize() + ".")
    return " ".join(out)


def weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Report current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
            },
        },
    }


def numeric_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "transfer_money",
                "description": "Send money between two accounts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "integer"},
                        "to_account": {"type": "integer"},
                        "amount": {"type": "number"},
                        "confirmed": {"type": "boolean"},
                    },
                    "required": [
                        "from_account", "to_account", "amount", "confirmed",
                    ],
                },
            },
        },
    ]


def shipping_tool():
    return {
        "type": "function",
        "function": {
            "name": "create_shipment",
            "description": "Book a shipment with a package list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["street", "zip"],
                    },
                    "express": {"type": "boolean"},
                },
                "required": ["items", "address"],
            },
        },
    }


PICK_NAMES = [
    "search_flights", "book_restaurant", "translate_text", "play_music",
    "set_alarm", "order_groceries", "convert_currency", "check_calendar",
]


def pick_tools():
    out = []
    for name in PICK_NAMES:
        a, b = name.split("_", 1)
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{a.title()} a {b} for the user.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        })
    return out


def generic_tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Log a {name.replace('_', ' ')} note.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


PICK_DETAILS = {
    "search_flights": "from Lisbon to Porto tomorrow morning, one passenger",
    "book_restaurant": "at Luigi's in Lisbon, tomorrow 7pm, two people",
    "translate_text": "'good morning' from English into Portuguese",
    "play_music": "the album Journal Replay on the kitchen speaker",
    "set_alarm": "for 6:40 tomorrow morning",
    "order_groceries": "two bags of rice and a litre of olive oil, deliver "
                       "to my saved home address",
    "convert_currency": "500 EUR into USD at today's rate",
    "check_calendar": "for this Wednesday afternoon, tell me the first slot",
}


def base_cases(rng):
    """One instance per grammar family, before --n repetition."""
    city = rng.choice(
        ["Lisbon", "Kyoto", "Nairobi", "Reykjavik", "Wellington", "Quito"])
    unit = rng.choice(["celsius", "fahrenheit"])
    a, b = rng.sample(range(1000, 9999), 2)
    amt = round(rng.uniform(5, 500), 2)
    items = rng.sample(
        ["umbrella", "laptop", "cheese", "rope", "lantern", "books"], 3)
    street = f"{rng.randint(1, 999)} {rng.choice(WORDS).title()} Street"
    zipc = f"{rng.randint(10000, 99999)}"
    pick = rng.choice(PICK_NAMES)
    cases = [
        {
            "family": "string_enum",
            "tools": [weather_tool()], "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          f"What is the weather in {city} in {unit}?"}],
            "expect": {"name": "get_weather",
                       "args": {"city": "str", "unit": ("enum",
                                                        ["celsius",
                                                         "fahrenheit"])}},
        },
        {
            "family": "numeric_bool",
            "tools": numeric_tools(), "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          f"Transfer {amt} from account {a} to account {b}, "
                          "and yes it is confirmed."}],
            "expect": {"name": "transfer_money",
                       "args": {"from_account": "int", "to_account": "int",
                                "amount": "num", "confirmed": "bool"}},
        },
        {
            "family": "array_nested",
            "tools": [shipping_tool()], "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          "Ship " + ", ".join(items[:-1]) + " and "
                          + items[-1] + f" to {street}, zip {zipc}."}],
            "expect": {"name": "create_shipment",
                       "args": {"items": ("list_of", "str"),
                                "address": ("keys", ["street", "zip"])}},
        },
        {
            "family": "pick_from_many",
            "tools": pick_tools(), "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          f"Call the {pick} tool right now, no questions: "
                          f"{PICK_DETAILS[pick]}."}],
            "expect": {"name": pick, "args": {"query": "str"}},
        },
        {
            "family": "two_calls",
            "tools": [generic_tool("save_note"),
                      generic_tool("send_receipt")],
            "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          "In one go: save the note 'red hatchback rented' "
                          "and send the receipt for it."}],
            "expect": {"names": ["save_note", "send_receipt"],
                       "args": {"text": "str"}},
        },
        {
            "family": "multi_handoff",
            "tools": [generic_tool("lookup_item"),
                      generic_tool("confirm_order"),
                      generic_tool("save_note")],
            "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": "Do you have the brass kettle?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "callh1", "type": "function",
                    "function": {"name": "lookup_item",
                                 "arguments": json.dumps(
                                     {"text": "brass kettle"})}}]},
                {"role": "tool", "tool_call_id": "callh1",
                 "content": "stock: 1"},
                {"role": "assistant",
                 "content": "One brass kettle is in stock."},
                {"role": "user", "content": "Then confirm the order."},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "callh2", "type": "function",
                    "function": {"name": "confirm_order",
                                 "arguments": json.dumps(
                                     {"text": "brass kettle"})}}]},
                {"role": "tool", "tool_call_id": "callh2",
                 "content": "order 4471 placed"},
                {"role": "assistant", "content": "Order 4471 is placed."},
                {"role": "user",
                 "content": "Save a final note with the order number."},
            ],
            "expect": {"name": "save_note", "args": {"text": "str"}},
        },
    ]
    brace_payload = {"cfg": {"emblem": "café 例",
                             "html": '<div class="a"></div>'},
                     "note": "x{y} → télé 例"}
    brace_doc = (
        "Copy this snippet verbatim, character by character, into the code "
        "argument of a save_snippet tool call: "
        + json.dumps(brace_payload, ensure_ascii=False)
    )
    cases.append({
        "family": "braces_unicode",
        "tools": [{
            "type": "function",
            "function": {
                "name": "save_snippet",
                "description": "Store a code snippet string.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": brace_doc}],
        "expect": {"name": "save_snippet", "args": {"code": "str"}},
    })
    big = " ".join(rng.choice(WORDS) for _ in range(400))
    cases.append({
        "family": "long_arguments",
        "tools": [generic_tool("save_note")], "tool_choice": "auto",
        "messages": [{"role": "user", "content":
                      "Save a note whose text is exactly this paragraph, do "
                      "not summarize it: " + big}],
        "expect": {"name": "save_note", "args": {"text": "str"}},
    })
    long_doc = filler(rng, 15000)
    cases.append({
        "family": "long_context",
        "tools": [weather_tool()], "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": "You are a careful assistant. "
                                          + long_doc},
            {"role": "user", "content":
             f"After all that context: weather in {city}, in {unit}."},
        ],
        "expect": {"name": "get_weather",
                   "args": {"city": "str", "unit": ("enum",
                                                    ["celsius",
                                                     "fahrenheit"])}},
    })
    return cases


def build_cases(salt, n, only, deep_words=0):
    rng = random.Random(salt)
    base = base_cases(rng)
    if deep_words:
        doc = filler(rng, int(deep_words))
        for c in base:
            if c["family"] == "long_context":
                c["messages"] = [
                    {"role": "system",
                     "content": "You are a careful assistant. " + doc},
                    c["messages"][1],
                ]
    out = []
    for i in range(n):
        for c in base:
            if only and c["family"] not in only:
                continue
            c = dict(c)
            c["id"] = f"{c['family']}-{i}"
            out.append(c)
    return out


def check_value(value, spec):
    if spec == "str":
        return isinstance(value, str) and value != ""
    if spec == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec == "num":
        return isinstance(value, (int, float)) and not isinstance(
            value, bool)
    if spec == "bool":
        return isinstance(value, bool)
    if isinstance(spec, tuple) and spec[0] == "enum":
        return value in spec[1]
    if isinstance(spec, tuple) and spec[0] == "list_of":
        return isinstance(value, list) and len(value) > 0 and all(
            check_value(v, spec[1]) for v in value)
    if isinstance(spec, tuple) and spec[0] == "keys":
        return isinstance(value, dict) and all(
            k in value and value[k] != "" for k in spec[1])
    return False


def evaluate(case, reply):
    """Full verdict for one case given its expectation block."""
    try:
        data = json.loads(reply)
    except (ValueError, TypeError):
        return "HTTP_ERR", None
    choices = data.get("choices") or []
    if not choices:
        return "HTTP_ERR", None
    calls = (choices[0].get("message") or {}).get("tool_calls") or []
    if not calls:
        if choices[0].get("finish_reason") == "length":
            return "TRUNCATED", None
        return "NO_CALL", None
    want_names = ([case["expect"]["name"]] if "name" in case["expect"]
                  else case["expect"]["names"])
    got = []
    for c in calls:
        fn = (c.get("function") or {})
        got.append((fn.get("name"), fn.get("arguments")))
    names_ok = [g for g in got if g[0] in want_names]
    if not names_ok:
        return "WRONG_NAME", got
    missing = set(want_names) - {g[0] for g in names_ok}
    if missing:
        return "MISSING_CALL", got
    for name, raw in names_ok:
        try:
            args = json.loads(raw)
        except (ValueError, TypeError):
            return "BAD_JSON", got
        if not isinstance(args, dict):
            return "BAD_JSON", got
        for key, spec in case["expect"]["args"].items():
            if key not in args:
                return "MISSING_ARG", got
            if not check_value(args[key], spec):
                return "TYPE_FAIL", got
    return "OK", got


def _stream_message(resp):
    """Reassemble an SSE chat stream into (message, finish_reason).

    Delta semantics mirror OpenAI: content appends, tool_calls carry an index
    and partial name/argument fragments that must be stitched. A frontend
    parser that mishandles a tag split across chunks shows up here as a
    message that never validates, exactly like in a real client.
    """
    content = ""
    finish = None
    acc = {}
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        choices = d.get("choices") or []
        if not choices:
            continue
        finish = choices[0].get("finish_reason") or finish
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            content += delta["content"]
        for tcd in delta.get("tool_calls") or []:
            slot = acc.setdefault(
                tcd.get("index", 0),
                {"id": None, "type": "function",
                 "function": {"name": "", "arguments": ""}})
            if tcd.get("id"):
                slot["id"] = tcd["id"]
            fn = tcd.get("function") or {}
            if isinstance(fn.get("name"), str):
                slot["function"]["name"] += fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["function"]["arguments"] += fn["arguments"]
    msg = {"role": "assistant", "content": content}
    if acc:
        msg["tool_calls"] = [acc[k] for k in sorted(acc)]
    return {"choices": [{"message": msg, "finish_reason": finish}]}, finish


def run_case(base_url, model, case, timeout, max_tokens, stream=False):
    body = json.dumps({
        "model": model,
        "messages": case["messages"],
        "tools": case["tools"],
        "tool_choice": case["tool_choice"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        if stream:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data, _ = _stream_message(r)
            reply = json.dumps(data)
        else:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                reply = r.read().decode()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        reply = ""
        print(f"  !! {case['id']}: {exc}", file=sys.stderr)
    verdict, detail = evaluate(case, reply) if reply else ("HTTP_ERR", None)
    preview = ""
    if verdict != "OK" and reply:
        try:
            msg = (json.loads(reply).get("choices") or [{}])[0].get(
                "message") or {}
            preview = (msg.get("content") or "")[:160]
        except (ValueError, TypeError):
            preview = reply[:160]
    return {"id": case["id"], "family": case["family"], "verdict": verdict,
            "s": round(time.perf_counter() - t0, 2), "got": detail,
            "content": preview}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--salt", type=int, required=True,
                    help="same salt means the same prompts, the bench.py law")
    ap.add_argument("--n", type=int, default=2,
                    help="repetitions of the seven-family set")
    ap.add_argument("--only", default="",
                    help="comma list of families to run")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="budget must clear the hidden reasoning phase plus "
                    "the call; 256 truncated reasoning to death on pick cases")
    ap.add_argument("--min-valid", type=float, default=100.0,
                    help="exit non-zero below this OK rate, percent")
    ap.add_argument("--deep-tokens", type=int, default=0,
                    help="replace the long_context filler with this many "
                    "tokens (~words x 1.3); 650000 reproduces a session that "
                    "must call a tool at 900k depth. One prefill each.")
    ap.add_argument("--stream", action="store_true",
                    help="exercise the SSE path: deltas reassembled client-"
                    "side, which is where a streaming parser bug surfaces")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    listed = json.loads(urllib.request.urlopen(
        args.base_url.rstrip("/") + "/v1/models", timeout=30).read())
    model = args.model or listed["data"][0]["id"]
    only = set(f.strip() for f in args.only.split(",")) if args.only else None
    cases = build_cases(args.salt, args.n, only,
                        int(args.deep_tokens / 1.3) if args.deep_tokens
                        else 0)
    print(f"tool-call battery: {len(cases)} cases, salt {args.salt}, "
          f"greedy, conc {args.conc}, model {model}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.conc) as ex:
        rows = list(ex.map(
            lambda c: run_case(args.base_url, model, c, args.timeout,
                               args.max_tokens, args.stream),
            cases))

    order = ["OK", "NO_CALL", "TRUNCATED", "BAD_JSON", "WRONG_NAME",
             "MISSING_CALL", "MISSING_ARG", "TYPE_FAIL", "HTTP_ERR"]
    families = sorted({r["family"] for r in rows})
    print(f"{'family':>16}  {'ok':>4}  " +
          "  ".join(f"{v[:8]:>8}" for v in order[1:]))
    for fam in families:
        frs = [r for r in rows if r["family"] == fam]
        cnt = {v: sum(1 for r in frs if r["verdict"] == v) for v in order}
        print(f"{fam:>16}  {cnt['OK']:>3}/{len(frs):<2}  " +
              "  ".join(f"{cnt[v]:>8}" for v in order[1:] if cnt[v]))
    ok = sum(1 for r in rows if r["verdict"] == "OK")
    rate = 100.0 * ok / len(rows) if rows else 0.0
    print(f"\nvalid tool-call rate: {ok}/{len(rows)} = {rate:.1f}%")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"salt": args.salt, "model": model, "rate": rate,
                       "rows": rows}, f, indent=1)
        print(f"wrote {args.json}")
    bad = [r for r in rows if r["verdict"] != "OK"]
    for r in bad:
        print(f"  FAIL {r['id']}: {r['verdict']} got={r['got']} "
              f"content={r['content']!r}")
    if rate < args.min_valid:
        sys.exit(1)


# keep the module importable for a fake-server selftest later
if __name__ == "__main__":
    main()
