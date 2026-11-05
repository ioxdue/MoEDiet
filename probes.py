"""Small, inspectable probe bank. No generator model or network calls."""

import json
from pathlib import Path

CONTROL = "__control__"
SPLITS = ("profile", "select", "audit")
# Two distinct situations per split. The same stems are used for target and control.
STEMS = (
    "Explain the mechanism involved.", "Give a concrete solution and justify it.",
    "What would you check first, and why?", "Work through a small example.",
    "Explain what happens next and why.", "Show how you would verify your answer.",
)
CASES = {
    "python": ("A mutable default list accumulates items between calls.", "Read a large CSV lazily using a generator.",
               "A closure in a loop returns the last index every time.", "Group log entries using collections.defaultdict.",
               "Two names refer to the same nested list after a shallow copy.", "Test an asynchronous context manager."),
    "c++": ("A std::vector reallocation invalidates an iterator.", "Transfer sole ownership using std::unique_ptr.",
            "A lambda captures a local variable by reference and escapes.", "Implement a move constructor for a buffer.",
            "A std::string_view outlives its backing string.", "Test destruction through a polymorphic base pointer."),
    "devops": ("A rolling deployment serves traffic before readiness checks pass.", "Roll back a failed deployment without losing logs.",
               "A deployment succeeds in staging but fails its production health check.", "Choose a threshold for a canary rollback.",
               "A Kubernetes pod is repeatedly OOMKilled.", "Restore a backup in isolation and verify the restored data."),
    "docker": ("A source change invalidates every Docker build layer.", "Produce a small runtime image with a multi-stage Dockerfile.",
               "A file copied into an image disappears under a bind mount.", "Configure a container health check.",
               "A PID 1 shell does not forward SIGTERM to its child.", "Verify that a named volume survives container replacement."),
    "postgresql": ("An index is ignored on a low-selectivity column.", "Inspect a slow query with EXPLAIN ANALYZE.",
                   "Two transactions acquire row locks in opposite order.", "Build a partial index for unprocessed jobs.",
                   "An old transaction prevents VACUUM from reclaiming tuples.", "Verify a retry after a serialization failure."),
    "sql": ("A LEFT JOIN loses unmatched rows after a WHERE filter.", "Rank sales totals within each region.",
            "A join duplicates each invoice total.", "Find customers with no orders using NOT EXISTS.",
            "COUNT(column) skips NULL values in a grouped query.", "Test a running total when two rows share a date."),
    "gardening": ("Leaves yellow after overwatering.", "Prepare soil for planting carrots.", "Seedlings bend toward a window.",
                  "Choose a watering schedule for a clay pot.", "A late frost damages new shoots.", "Check whether stored seeds will germinate."),
    "navigation": ("A compass bearing differs from true north.", "Estimate walking time using a contour map.", "Two map landmarks seem misaligned.",
                   "Convert a map scale into walking distance.", "Wind pushes a boat away from its intended bearing.", "Check a position using two bearings."),
    "music": ("Two strings produce beats when played together.", "Transpose a melody to a new key.", "A chord sounds unstable before resolving.",
              "Count the pulses in six-eight time.", "Shortening a vibrating string raises its pitch.", "Check whether a recording is in tune."),
}
ALIASES = {"cpp": "c++", "postgres": "postgresql"}
GENERAL = ("gardening", "navigation", "music")
NEUTRAL = ("A calendar spans a leap day.", "Divide a grocery bill between three friends.",
           "Two clocks disagree after a battery change.", "Find the median of five distances.",
           "A queue is served in arrival order.", "Check a temperature conversion by reversing it.")
GENERIC = ("Introduce a basic concept with a worked example.", "Correct a common beginner mistake.",
           "Diagnose an unexpected outcome on a typical task.", "Compare two methods for a familiar problem.",
           "Predict an edge case in a realistic scenario.", "Design a reproducible check for a subtle error.")


def clean_tags(tags):
    names = [t.strip().casefold() for t in tags]
    names = list(dict.fromkeys(ALIASES.get(t, t) for t in names))
    if not names or any(t in ('', CONTROL) for t in names):
        raise ValueError(f"Serve almeno un tag non vuoto. {CONTROL} è riservato.")
    return names


def prompts_for(topic, split_index):
    cases = CASES.get(topic, GENERIC)
    return [f"Topic: {topic}. {cases[i]} {STEMS[i]}" for i in range(2 * split_index, 2 * split_index + 2)]


def make_probes(tags, custom=None):
    if custom:
        corpus = json.loads(Path(custom).read_text(encoding="utf-8"))
    else:
        controls = [topic for topic in GENERAL if topic not in tags]
        corpus = {}
        for i, split in enumerate(SPLITS):
            groups = {tag: prompts_for(tag, i) for tag in tags}
            groups[CONTROL] = [prompt for topic in controls for prompt in prompts_for(topic, i)] or [
                f"{NEUTRAL[j]} {STEMS[j]}" for j in range(2 * i, 2 * i + 2)]
            corpus[split] = groups
    if not isinstance(corpus, dict) or set(corpus) != set(SPLITS):
        raise ValueError(f"Il corpus richiede esattamente questi split: {SPLITS}")
    seen = set()
    for split, groups in corpus.items():
        if not isinstance(groups, dict) or set(groups) != {*tags, CONTROL}:
            raise ValueError(f"Gruppi errati in {split}. Servono tutti i tag e {CONTROL}.")
        for group, prompts in groups.items():
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(f"Nessuna sonda in {split}/{group}. Fornire un corpus con esempi distinti.")
            for prompt in prompts:
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Sonda vuota o non testuale in {split}/{group}.")
                key = ' '.join(prompt.casefold().split())
                if key in seen:
                    raise ValueError(f"Sonda duplicata nel corpus: {prompt!r}")
                seen.add(key)
    return corpus
