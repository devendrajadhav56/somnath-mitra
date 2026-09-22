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

    print("=" * 92)
    print(f"heuristic pass: {passed}/{len(PROBES)}  (eyeball the replies — heuristics are a signal, not proof)")
    return passed == len(PROBES)


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
