"""Term expansion — domain-specific synonym/abbreviation mappings."""

from __future__ import annotations

import re

TERM_EXPANSIONS: dict[str, list[str]] = {
    "auth": ["authentication", "authorization", "login", "token", "oauth"],
    "authn": ["authentication", "identity", "credential"],
    "authz": ["authorization", "permission", "access control", "rbac"],
    "api": ["endpoint", "rest", "http", "interface"],
    "config": ["configuration", "settings", "environment", "setup"],
    "deploy": ["deployment", "release", "rollout", "infrastructure"],
    "db": ["database", "storage", "persistence", "sql"],
    "perf": ["performance", "latency", "throughput", "benchmark"],
    "error": ["exception", "failure", "bug", "issue", "traceback"],
    "install": ["installation", "setup", "dependency", "package"],
    "k8s": ["kubernetes", "cluster", "pod", "container", "helm"],
    "ml": ["machine learning", "model", "training", "inference"],
    "llm": ["large language model", "gpt", "claude", "inference", "prompt"],
    # Music genre terms
    "hip-hop": ["rap", "hip hop", "emcee", "mc", "dj", "breakbeat"],
    "hiphop": ["rap", "hip hop", "emcee", "mc", "dj", "breakbeat"],
    "blues": ["delta blues", "chicago blues", "electric blues", "twelve-bar"],
    "jazz": ["bebop", "swing", "fusion", "improvisation", "big band"],
    "metal": ["heavy metal", "doom", "thrash", "black metal", "riff"],
    "punk": ["hardcore", "new wave", "post-punk", "diy", "three-chord"],
    "soul": ["rhythm and blues", "rnb", "gospel", "motown", "stax"],
    "electronic": ["synthesizer", "drum machine", "techno", "house", "edm"],
    "country": ["honky-tonk", "nashville", "bluegrass", "outlaw country"],
}


def expand_terms(query_lower: str) -> list[str]:
    """Return expansion terms for abbreviations/jargon found in *query_lower*."""
    words = re.findall(r"\w+", query_lower)
    expansions: list[str] = []
    seen: set[str] = set(words)
    for word in words:
        for term in TERM_EXPANSIONS.get(word, []):
            if term not in seen:
                expansions.append(term)
                seen.add(term)
    return expansions
