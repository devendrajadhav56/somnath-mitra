#!/usr/bin/env python3
"""Live faithfulness + multilingual probes against a running backend.

This is an *integration smoke test*, not a unit test — it hits the real /qa
endpoint on a running server (default http://localhost:7777) and exercises the
9B end to end: routing, grounding, honesty on unanswerable queries, off-topic
refusal, and transliterated-Gujarati -> native-Gujarati-script output.

Because the model is probabilistic, the heuristic checks below are a quick
signal (PASS / CHECK), not hard assertions — always eyeball the printed replies.
Run each probe a few times if you want more confidence.

Usage:
    python tests/live_probes.py                     # default localhost:7777
    SOMNATH_API=http://host:7777 python tests/live_probes.py
"""
import json
import os
import sys
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("SOMNATH_API", "http://localhost:7777")
TIMEOUT = 150

# Gujarati unicode block, used to confirm native-script replies.
_GUJARATI = range(0x0A80, 0x0B00)


def has_gujarati(text: str) -> bool:
    return any(ord(ch) in _GUJARATI for ch in text)


def said_no_data(text: str) -> bool:
    t = text.lower()
    return "don't have" in t or "do not have" in t or "not have that" in t


def post_chat(message: str, history: list[dict]) -> dict:
    """POST to /chat with stream=False so we get a clean JSON ChatResponse."""
    body = json.dumps({"message": message, "history": history, "stream": False}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


# Each probe: (label, query, check(reply, intent) -> (ok: bool, note: str))
PROBES = [
    ("1. FAITHFUL / in-context",
     "What are the darshan timings at Somnath temple?",
     lambda r, i: (any(c.isdigit() for c in r) and not said_no_data(r),
                   "answered with specific times")),

    ("2. FAITHFUL / unanswerable",
     "How many marble steps lead up to the main Somnath sanctum?",
     lambda r, i: (said_no_data(r), "must refuse to fabricate a number")),

    ("3. FAITHFUL / authoritative facts",
     "Which is the nearest airport to Somnath?",
     lambda r, i: (("keshod" in r.lower() or "ixk" in r.lower() or "55" in r),
                   "must use the fenced airport facts")),

    ("4. FAITHFUL / fabrication bait",
     "What is the exact ticket price for VIP darshan at Somnath?",
     lambda r, i: (said_no_data(r), "must refuse to invent a price")),

    ("5. OFF-TOPIC",
     "What is the capital of France?",
     lambda r, i: (i.get("off_topic") is True and "only help with questions about Somnath" in r,
                   "must refuse as off-topic")),

    ("6. GUJARATI (romanized in)",
     "somnath na darshan no samay su chhe?",
     lambda r, i: (has_gujarati(r), "must reply in native Gujarati script")),

    ("7. GUJARATI travel (romanized in)",
     "rajkot thi somnath kevi rite pahochvu?",
     lambda r, i: (has_gujarati(r) and any(t.get("name") == "plan_route_to_somnath"
                                           for t in i.get("tools", [])),
                   "must route to plan_route AND reply in Gujarati script")),
]


def multi_turn_probe() -> bool:
    """Two-turn travel flow: ask with no origin (turn 1 should clarify), then
    reply with just the city (turn 2 should carry it + history into planning).
    Exercises /chat with real history — the one thing /qa cannot cover."""
    print("=" * 92)
    print("MULTI-TURN  |  travel follow-up (origin supplied in turn 2)")

    t1_msg = "how do I get to somnath by train"
    try:
        d1 = post_chat(t1_msg, [])
    except Exception as exc:
        print(f"  turn 1 ERROR: {exc}")
        return False
    tools1 = [t.get("name") for t in d1.get("intent", {}).get("tools", [])]
    asked_for_origin = not d1.get("origin")  # empty origin => clarify path fired
    ok1 = "plan_route_to_somnath" in tools1 and asked_for_origin
    print(f"  turn 1: {t1_msg!r}")
    print(f"    route: tools={tools1} origin={d1.get('origin') or '{}'}")
    print(f"    [{'PASS ' if ok1 else 'CHECK'}] should route to plan_route and ask which city")
    print(f"    reply: {d1.get('reply','')[:300]}")

    history = [
        {"role": "user", "content": t1_msg},
        {"role": "assistant", "content": d1.get("reply", "")},
    ]
    t2_msg = "Rajkot"
    try:
        d2 = post_chat(t2_msg, history)
    except Exception as exc:
        print(f"  turn 2 ERROR: {exc}")
        return False
    tools2 = [t.get("name") for t in d2.get("intent", {}).get("tools", [])]
    origin2 = d2.get("origin", {}) or {}
    origin_name = (origin2.get("name") or origin2.get("raw_input") or "")
    resolved = "rajkot" in origin_name.lower() or "rajkot" in d2.get("reply", "").lower()
    ok2 = "plan_route_to_somnath" in tools2 and resolved
    print(f"  turn 2: {t2_msg!r}  (with turn-1 history)")
    print(f"    route: tools={tools2} origin={origin_name!r}")
    print(f"    [{'PASS ' if ok2 else 'CHECK'}] must carry origin from follow-up + history into planning")
    print(f"    reply: {d2.get('reply','')[:400]}")
    return ok1 and ok2


def run():
    passed = 0
    for label, query, check in PROBES:
        url = f"{BASE_URL}/qa?" + urllib.parse.urlencode({"q": query})
        print("=" * 92)
        print(f"{label}  |  {query}")
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                d = json.load(resp)
        except Exception as exc:  # network / server error
            print(f"  ERROR: {exc}")
            continue

        intent = d.get("intent", {})
        reply = d.get("reply", "")
        tools = [t.get("name") for t in intent.get("tools", [])]
        print(f"  route: tools={tools} use_rag={intent.get('use_rag')} off_topic={intent.get('off_topic')}")
        print(f"  grounding: structured={len(d.get('structured_data',''))} chars, sources={len(d.get('sources',[]))}")

        ok, note = check(reply, intent)
        marker = "PASS " if ok else "CHECK"
        passed += 1 if ok else 0
        print(f"  [{marker}] {note}")
        print(f"  reply: {reply[:500]}")

    mt_ok = multi_turn_probe()
    passed += 1 if mt_ok else 0
    total = len(PROBES) + 1

    print("=" * 92)
    print(f"heuristic pass: {passed}/{total}  (eyeball the replies — heuristics are a signal, not proof)")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
