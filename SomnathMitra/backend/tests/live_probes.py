#!/usr/bin/env python3
"""Live probe matrix for the Somnath backend — faithfulness, routing, links,
travel, local search, and all four language modes, single- and multi-turn.

This is an *integration smoke test*, not a unit test: it hits the real /qa and
/chat endpoints on a running server and exercises the 9B end to end. Because the
model is probabilistic, the per-probe checks are a quick PASS/CHECK signal, not
hard assertions — always eyeball the printed replies. Run probes a few times if
you want more confidence.

Usage:
    python tests/live_probes.py                 # run everything (~10+ min, many LLM calls)
    python tests/live_probes.py travel links    # only these categories
    python tests/live_probes.py --list          # list categories
    SOMNATH_API=http://host:7777 python tests/live_probes.py lang

Categories: temple, links, local, travel, faithful, offtopic, lang, multiturn
"""
import json
import os
import sys
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("SOMNATH_API", "http://localhost:7777")
TIMEOUT = 150

GUESTHOUSE_URL = "somnath.org/guesthouse"
DONATION_URL = "somnath.org/online-donation"

_GUJARATI = range(0x0A80, 0x0B00)
_DEVANAGARI = range(0x0900, 0x0980)


def has_gujarati(t: str) -> bool:
    return any(ord(c) in _GUJARATI for c in t)


def has_devanagari(t: str) -> bool:
    return any(ord(c) in _DEVANAGARI for c in t)


def said_no_data(t: str) -> bool:
    t = t.lower()
    return "don't have" in t or "do not have" in t or "not have that" in t


# ── HTTP ──────────────────────────────────────────────────────────────────────

def get_qa(query: str) -> dict:
    url = f"{BASE_URL}/qa?" + urllib.parse.urlencode({"q": query})
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.load(r)


def post_chat(message: str, history: list[dict]) -> dict:
    body = json.dumps({"message": message, "history": history, "stream": False}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


# ── Check builders: each returns check(reply, intent) -> (ok: bool, note: str) ──

def _mk(fn, note):
    return lambda r, i: (bool(fn(r, i)), note)


def route(name):
    return _mk(lambda r, i: any(t.get("name") == name for t in i.get("tools", [])),
               f"routes to {name}")


def rag_only():
    return _mk(lambda r, i: i.get("use_rag") is True and not i.get("tools"),
               "RAG-only (no tools)")


def link(u, what):
    return _mk(lambda r, i: u in r.lower(), f"includes {what} link")


def kw(words, note):
    return _mk(lambda r, i: any(w in r.lower() for w in words), note)


def script(textfn, what):
    return _mk(lambda r, i: textfn(r), f"reply in {what}")


def no_data():
    return _mk(lambda r, i: said_no_data(r), "refuses to fabricate")


def is_offtopic():
    return _mk(lambda r, i: i.get("off_topic") is True, "off-topic refusal")


def answers():
    return _mk(lambda r, i: len(r.strip()) > 0 and not i.get("off_topic"), "answers (not refused)")


def AND(*checks):
    def run_all(r, i):
        ok, notes = True, []
        for c in checks:
            o, n = c(r, i)
            ok = ok and o
            notes.append(n)
        return ok, " + ".join(notes)
    return run_all


TIMINGS = kw([":", "am", "pm", "६", "૬", "morning", "evening"], "gives a time")

# ── Single-turn probes: (category, query, check) ───────────────────────────────

SINGLE_TURN = [
    # temple info / RAG
    ("temple", "What are the darshan timings at Somnath temple?", AND(route("get_temple_info"), TIMINGS)),
    ("temple", "What time is the aarti at Somnath?", TIMINGS),
    ("temple", "What is the dress code and are mobile phones allowed inside Somnath temple?",
     kw(["dress", "footwear", "shoe", "mobile", "phone", "photograph"], "mentions rules")),
    ("temple", "Tell me about prasad at Somnath temple", route("get_temple_info")),
    ("temple", "Tell me about the history and significance of Somnath temple", answers()),

    # link-injection behaviors
    ("links", "Where can I stay near Somnath temple?", link(GUESTHOUSE_URL, "guesthouse booking")),
    ("links", "I want to book a room at the Somnath guest house", link(GUESTHOUSE_URL, "guesthouse booking")),
    ("links", "I want to book a pooja online at Somnath", link(DONATION_URL, "pooja/donation")),
    ("links", "How can I donate to the Somnath temple?", link(DONATION_URL, "donation")),

    # local structured search
    ("local", "restaurants near Somnath temple", route("search_restaurants")),
    ("local", "pure vegetarian food near Somnath", route("search_restaurants")),
    ("local", "nearest hospital to Somnath", route("search_hospitals")),
    ("local", "medical store / pharmacy near Somnath", route("search_pharmacies")),
    ("local", "places to visit near Somnath temple", route("search_pois")),
    ("local", "price of sarees at the Somnath temple shop", route("search_shop")),

    # travel spread
    ("travel", "trains from Ahmedabad to Somnath", AND(route("plan_route_to_somnath"),
                                                       kw(["train", "veraval", "somnath", "station"], "train info"))),
    ("travel", "trains from Mumbai to Somnath", route("plan_route_to_somnath")),
    ("travel", "how do I reach Somnath from Delhi?", route("plan_route_to_somnath")),
    ("travel", "buses from Rajkot to Somnath", route("plan_route_to_somnath")),
    ("travel", "are there flights to Somnath?", AND(route("plan_route_to_somnath"),
                                                    kw(["airport", "keshod", "diu", "no direct", "road"], "flight/airport reality"))),
    ("travel", "which is the nearest airport to Somnath?", kw(["keshod", "ixk", "55"], "uses airport facts")),
    ("travel", "should I get down at Veraval Junction or Somnath station?", rag_only()),

    # faithfulness
    ("faithful", "How many marble steps lead up to the main Somnath sanctum?", no_data()),
    ("faithful", "What is the exact ticket price for VIP darshan at Somnath?", no_data()),

    # off-topic
    ("offtopic", "What is the capital of France?", is_offtopic()),

    # language modes
    ("lang", "सोमनाथ मंदिर के दर्शन का समय क्या है?", script(has_devanagari, "Hindi/Devanagari")),
    ("lang", "સોમનાથ મંદિરના દર્શનનો સમય શું છે?", script(has_gujarati, "Gujarati")),
    ("lang", "somnath na darshan no samay su chhe?", script(has_gujarati, "native Gujarati (from romanized)")),
    ("lang", "somnath mandir ke darshan ka samay kya hai?", answers()),
]

# ── Multi-turn conversations: (category, name, [(query, check), ...]) ───────────

CONVERSATIONS = [
    ("multiturn", "travel: origin supplied in turn 2", [
        ("how do I get to Somnath by train?", route("plan_route_to_somnath")),
        ("Rajkot", AND(route("plan_route_to_somnath"), kw(["rajkot"], "plans from Rajkot"))),
    ]),
    ("multiturn", "travel: far origin over two turns", [
        ("I want to travel to Somnath", answers()),
        ("from Mumbai", AND(route("plan_route_to_somnath"), kw(["mumbai", "veraval", "train", "bus"], "plans from Mumbai"))),
    ]),
    ("multiturn", "topic switch: timings then dress code", [
        ("what are the darshan timings?", TIMINGS),
        ("and what is the dress code?", kw(["dress", "footwear", "shoe", "cloth", "attire"], "answers dress code")),
    ]),
    ("multiturn", "origin switch mid-conversation", [
        ("trains from Rajkot to Somnath", AND(route("plan_route_to_somnath"), kw(["rajkot"], "from Rajkot"))),
        ("what about from Ahmedabad?", AND(route("plan_route_to_somnath"), kw(["ahmedabad"], "switches to Ahmedabad"))),
    ]),
    ("multiturn", "restaurant follow-up", [
        ("restaurants near Somnath", route("search_restaurants")),
        ("any pure veg ones?", route("search_restaurants")),
    ]),
]

CATEGORIES = ["temple", "links", "local", "travel", "faithful", "offtopic", "lang", "multiturn"]


# ── Runner ────────────────────────────────────────────────────────────────────

def _print_result(ok, note, reply):
    print(f"    [{'PASS ' if ok else 'CHECK'}] {note}")
    print(f"    reply: {reply[:400]}")


def run(categories=None):
    passed = total = 0

    for cat, query, check in SINGLE_TURN:
        if categories and cat not in categories:
            continue
        total += 1
        print("=" * 92)
        print(f"[{cat}]  {query}")
        try:
            d = get_qa(query)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue
        intent, reply = d.get("intent", {}), d.get("reply", "")
        tools = [t.get("name") for t in intent.get("tools", [])]
        print(f"    route: tools={tools} use_rag={intent.get('use_rag')} off_topic={intent.get('off_topic')}")
        ok, note = check(reply, intent)
        passed += 1 if ok else 0
        _print_result(ok, note, reply)

    for cat, name, turns in CONVERSATIONS:
        if categories and cat not in categories:
            continue
        total += 1
        print("=" * 92)
        print(f"[{cat}]  {name}")
        history, convo_ok = [], True
        for idx, (msg, check) in enumerate(turns, 1):
            try:
                d = post_chat(msg, history)
            except Exception as exc:
                print(f"    turn {idx} ERROR: {exc}")
                convo_ok = False
                break
            intent, reply = d.get("intent", {}), d.get("reply", "")
            tools = [t.get("name") for t in intent.get("tools", [])]
            origin = (d.get("origin") or {})
            print(f"    turn {idx}: {msg!r}")
            print(f"      route: tools={tools} origin={origin.get('name') or origin.get('raw_input') or '{}'}")
            ok, note = check(reply, intent)
            convo_ok = convo_ok and ok
            print(f"      [{'PASS ' if ok else 'CHECK'}] {note}")
            print(f"      reply: {reply[:300]}")
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        passed += 1 if convo_ok else 0

    print("=" * 92)
    print(f"heuristic pass: {passed}/{total}  (eyeball the replies — heuristics are a signal, not proof)")
    return passed == total


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        print("categories:", ", ".join(CATEGORIES))
        sys.exit(0)
    cats = set(args) if args else None
    if cats:
        unknown = cats - set(CATEGORIES)
        if unknown:
            print(f"unknown categories: {unknown}; valid: {CATEGORIES}")
            sys.exit(2)
    sys.exit(0 if run(cats) else 1)
