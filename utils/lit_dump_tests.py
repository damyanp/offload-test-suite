#!/usr/bin/env python3
"""Dump a JSON description of the tests lit would run, without running them.

Uses lit purely as a library; accepts all the normal lit arguments (test paths,
--param, --filter, --shard, ...) plus:

  --dump-json FILE   output file (default: stdout)
  --lit-path DIR     directory containing the 'lit' package (e.g.
                     llvm/utils/lit) if it is not already importable
  --assume F=BOOL    (repeatable) treat feature F as always true/false when
                     computing REQUIRES groups, e.g. --assume Vulkan=false

Each test's REQUIRES lines are combined (all expressions must hold) into a
single boolean function. Tests whose functions are logically equivalent share a
"REQUIRES group", e.g. `Int64 && Fp16` and `Fp16 && !(!Int64)` are one group.
Regular expressions ({{...}}) are not allowed in REQUIRES and are reported as
errors (the script then exits with status 1). Tests whose REQUIRES can never be
satisfied get requires_group "<never>" but are omitted from requires_groups.

Example:
  python lit_dump_tests.py --lit-path llvm/utils/lit --dump-json tests.json \
      -D foo=bar path/to/tests
"""

import argparse
import json
import os
import platform
import re
import sys

MAX_VARIABLES = 16
NEVER = "<never>"

_TOKEN = re.compile(r"\A\s*([()]|&&|\|\||!|[-+=._a-zA-Z0-9]+)\s*(.*)\Z", re.S)


# ---------------------------------------------------------------------------
# REQUIRES expression parsing (lit's BooleanExpression grammar, minus regexes)
#   or_expr  :: and_expr ('||' and_expr)*
#   and_expr :: not_expr ('&&' not_expr)*
#   not_expr :: '!' not_expr | '(' or_expr ')' | identifier
# AST nodes: ("var", name) | ("not", n) | ("and", a, b) | ("or", a, b)
# ---------------------------------------------------------------------------
def _tokenize(text):
    if "{{" in text:
        raise ValueError("regular expressions are not allowed in REQUIRES: %r" % text)
    tokens = []
    while text.strip():
        m = _TOKEN.match(text)
        if m is None:
            raise ValueError("couldn't parse text: %r" % text)
        tokens.append(m.group(1))
        text = m.group(2)
    return tokens


def parse_expression(text):
    tokens = _tokenize(text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def advance():
        nonlocal pos
        pos += 1

    def parse_not():
        tok = peek()
        if tok == "!":
            advance()
            return ("not", parse_not())
        if tok == "(":
            advance()
            node = parse_or()
            if peek() != ")":
                raise ValueError("expected ')' in expression: %r" % text)
            advance()
            return node
        if tok is None or tok in ("&&", "||", ")"):
            raise ValueError("expected '!', '(' or identifier in expression: %r" % text)
        advance()
        return ("var", tok)

    def parse_and():
        node = parse_not()
        while peek() == "&&":
            advance()
            node = ("and", node, parse_not())
        return node

    def parse_or():
        node = parse_and()
        while peek() == "||":
            advance()
            node = ("or", node, parse_and())
        return node

    node = parse_or()
    if peek() is not None:
        raise ValueError("unexpected %r in expression: %r" % (peek(), text))
    return node


def collect_variables(node, out):
    if node[0] == "var":
        if node[1] != "true":
            out.add(node[1])
    else:
        for child in node[1:]:
            collect_variables(child, out)


# ---------------------------------------------------------------------------
# Boolean functions as truth tables: an int whose bit i is the function value
# for the assignment where variable j is set iff bit j of i is set.
# ---------------------------------------------------------------------------
def truth_table(node, variables, assume=None):
    assume = assume or {}
    n = len(variables)
    size = 1 << n
    full = (1 << size) - 1
    masks = {}
    for j, name in enumerate(variables):
        bits = 0
        for i in range(size):
            if (i >> j) & 1:
                bits |= 1 << i
        masks[name] = bits

    def ev(nd):
        kind = nd[0]
        if kind == "var":
            name = nd[1]
            if name == "true":
                return full
            if name in assume:
                return full if assume[name] else 0
            return masks.get(name, 0)
        if kind == "not":
            return full ^ ev(nd[1])
        if kind == "and":
            return ev(nd[1]) & ev(nd[2])
        return ev(nd[1]) | ev(nd[2])

    return ev(node)


def relevant_variables(table, variables):
    n = len(variables)
    size = 1 << n
    result = []
    for j, name in enumerate(variables):
        depends = any(
            ((table >> i) & 1) != ((table >> (i ^ (1 << j))) & 1)
            for i in range(size)
        )
        if depends:
            result.append(name)
    return result


def minterms_of(table, n):
    return [i for i in range(1 << n) if (table >> i) & 1]


def prime_implicants(minterms, n):
    # An implicant is (value, mask): mask bits are "don't care".
    current = {(m, 0) for m in minterms}
    primes = set()
    while current:
        merged = set()
        used = set()
        by_mask = {}
        for imp in current:
            by_mask.setdefault(imp[1], []).append(imp)
        for mask, imps in by_mask.items():
            values = {v for v, _ in imps}
            for v in values:
                for j in range(n):
                    bit = 1 << j
                    if mask & bit or v & bit:
                        continue
                    if (v | bit) in values:
                        merged.add((v, mask | bit))
                        used.add((v, mask))
                        used.add((v | bit, mask))
        primes |= current - used
        current = merged
    return primes


def covers(imp, minterm):
    value, mask = imp
    return (minterm & ~mask) == value


def literal_count(imp, n):
    return n - bin(imp[1]).count("1")


def minimal_cover(minterms, n):
    primes = sorted(prime_implicants(minterms, n))
    remaining = set(minterms)
    chosen = []
    # Essential prime implicants first.
    for m in minterms:
        cands = [p for p in primes if covers(p, m)]
        if len(cands) == 1 and cands[0] not in chosen:
            chosen.append(cands[0])
    for p in chosen:
        remaining -= {m for m in remaining if covers(p, m)}
    # Then greedy: most newly covered minterms, fewest literals, deterministic.
    while remaining:
        best = max(
            (p for p in primes if p not in chosen),
            key=lambda p: (
                sum(1 for m in remaining if covers(p, m)),
                -literal_count(p, n),
                tuple(-x for x in p),
            ),
        )
        chosen.append(best)
        remaining -= {m for m in remaining if covers(best, m)}
    return chosen


def render_group(variables, table):
    """Minimal sum-of-products rendering in lit syntax."""
    n = len(variables)
    if n == 0:
        return "true" if table & 1 else NEVER
    minterms = minterms_of(table, n)
    terms = []
    for value, mask in minimal_cover(minterms, n):
        lits = []
        for j, name in enumerate(variables):
            if mask & (1 << j):
                continue
            lits.append(name if value & (1 << j) else "!" + name)
        terms.append(lits)
    terms.sort(key=lambda t: (len(t), [x.lstrip("!") + x for x in t]))
    return " || ".join(" && ".join(t) for t in terms)


def compute_requires_group(requires, assume=None):
    """Returns (key, display_name, features) for a list of REQUIRES strings.

    `assume` maps feature names to a fixed True/False value.
    """
    assume = assume or {}
    nodes = []
    for text in requires:
        for part in text.split(","):
            part = part.strip()
            if part:
                nodes.append(parse_expression(part))
    names = set()
    for node in nodes:
        collect_variables(node, names)
    names = sorted(n for n in names if n not in assume)
    if len(names) > MAX_VARIABLES:
        raise ValueError(
            "REQUIRES uses %d distinct features (max %d)" % (len(names), MAX_VARIABLES)
        )
    combined = ("var", "true")
    for node in nodes:
        combined = ("and", combined, node)
    table = truth_table(combined, names, assume)
    support = relevant_variables(table, names)
    # Irrelevant variables don't affect the result, so re-evaluate over the support
    # (truth_table treats variables outside its list as false).
    table = truth_table(combined, support, assume)
    return (tuple(support), table), render_group(support, table), support


# ---------------------------------------------------------------------------
# lit integration
# ---------------------------------------------------------------------------
def split_own_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dump-json", metavar="FILE")
    parser.add_argument("--lit-path", metavar="DIR")
    parser.add_argument("--assume", metavar="FEATURE=true|false", action="append", default=[])
    own, rest = parser.parse_known_args(argv)
    return own, rest


def parse_assumptions(items):
    assume = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or value.lower() not in ("true", "false"):
            raise SystemExit("error: --assume expects FEATURE=true|false, got %r" % item)
        assume[name.strip()] = value.lower() == "true"
    return assume


def describe_test(test, TestRunner):
    entry = {
        "name": test.getFullName(),
        "suite": test.suite.name,
        "path_in_suite": list(test.path_in_suite),
        "source_path": test.getSourcePath(),
        "requires": list(test.requires),
        "xfail": list(test.xfails),
        "unsupported": list(test.unsupported),
    }
    try:
        parsed = TestRunner._parseKeywords(test.getSourcePath(), require_script=False)
        entry["requires"] += parsed["REQUIRES:"] or []
        entry["xfail"] += parsed["XFAIL:"] or []
        entry["unsupported"] += parsed["UNSUPPORTED:"] or []
    except (ValueError, OSError, UnicodeDecodeError) as e:
        entry["error"] = str(e)
    return entry


def build_requires_groups(entries, assume=None):
    groups = {}
    for entry in entries:
        try:
            key, name, features = compute_requires_group(entry["requires"], assume)
        except ValueError as e:
            entry["requires_error"] = str(e)
            continue
        entry["requires_group"] = name
        if name == NEVER:
            continue
        group = groups.setdefault(
            key, {"group": name, "features": list(features), "tests": []}
        )
        group["tests"].append(entry["name"])
    result = []
    for group in sorted(groups.values(), key=lambda g: (len(g["features"]), g["group"])):
        group["test_count"] = len(group["tests"])
        result.append(group)
    return result


def main():
    own, lit_args = split_own_args(sys.argv[1:])
    assume = parse_assumptions(own.assume)
    if own.lit_path:
        sys.path.insert(0, own.lit_path)

    import lit.cl_arguments
    import lit.discovery
    import lit.LitConfig
    import lit.main
    import lit.TestRunner

    # lit's parser reads sys.argv (and LIT_OPTS) itself.
    sys.argv = [sys.argv[0]] + lit_args
    opts = lit.cl_arguments.parse_args()

    lit_config = lit.LitConfig.LitConfig(
        progname=os.path.basename(sys.argv[0]),
        path=opts.path,
        diagnostic_level=opts.diagnostic_level,
        useValgrind=opts.useValgrind,
        valgrindLeakCheck=opts.valgrindLeakCheck,
        valgrindArgs=opts.valgrindArgs,
        noExecute=opts.noExecute,
        debug=opts.debug,
        isWindows=platform.system() == "Windows",
        order=opts.order,
        params=lit.main.create_params({}, opts.user_params),
        config_prefix=opts.configPrefix,
        pass_env=opts.pass_env,
        per_test_coverage=opts.per_test_coverage,
        gtest_sharding=opts.gtest_sharding,
        maxRetriesPerTest=opts.maxRetriesPerTest,
        update_tests=opts.update_tests,
        maxIndividualTestTime=opts.maxIndividualTestTime,
    )

    tests = lit.discovery.find_tests_for_inputs(lit_config, opts.test_paths)
    if not tests:
        sys.stderr.write("error: did not discover any tests for provided path(s)\n")
        return 2

    lit.main.determine_order(tests, opts.order)
    tests = [
        t
        for t in tests
        if opts.filter.search(t.getFullName())
        and not opts.filter_out.search(t.getFullName())
    ]
    if opts.filterFailed:
        tests = [t for t in tests if t.previous_failure]
    if opts.shard:
        run, shards = opts.shard
        tests = lit.main.filter_by_shard(tests, run, shards, lit_config)
    tests = tests[: opts.max_tests]

    entries = [describe_test(t, lit.TestRunner) for t in tests]
    groups = build_requires_groups(entries, assume)
    result = {"assumptions": assume, "requires_groups": groups, "tests": entries}

    if own.dump_json:
        with open(own.dump_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    else:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")

    status = 0
    for e in entries:
        if "requires_error" in e:
            sys.stderr.write("error: %s: %s\n" % (e["name"], e["requires_error"]))
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
