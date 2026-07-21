"""Expand experiments/corpus.json to ~200 passages and emit one QA item per passage.

Keeps the 15 curated passages unchanged (so existing pinned QA items stay valid),
then generates 185 new passages across the 8 agent-memory categories using
template families with slot variation. Each new passage is paired with a QA item
(question, answer, keywords, evidence_ids). Writes:
  experiments/corpus.json          (200 samples)
  experiments/agent_qa.json         (200 QA items, one per passage)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments" / "corpus.json"
QA_OUT = ROOT / "experiments" / "agent_qa.json"

random.seed(20260720)

# Preserved noun entities (never minified). QA answers are drawn from these.
LANGS = ["Python", "TypeScript", "JavaScript", "Rust", "Go", "Kotlin", "Swift",
         "Ruby", "PHP", "C++", "C#", "Scala", "Elixir", "Haskell", "Julia", "R", "Lua"]
DBS = ["PostgreSQL", "MySQL", "MongoDB", "Redis", "DynamoDB", "Cassandra",
       "Elasticsearch", "Neo4j", "ClickHouse", "Snowflake", "BigQuery", "SQLite"]
INFRA = ["Docker", "Kubernetes", "Nomad", "Terraform", "Ansible", "Pulumi",
         "Helm", "Argo CD", "Vault", "Consul", "Etcd", "Nginx", "Envoy", "Istio"]
CI = ["GitHub Actions", "GitLab CI", "Jenkins", "CircleCI", "Buildkite", "Spinnaker"]
OBS = ["Prometheus", "Grafana", "Datadog", "Splunk", "Loki", "Jaeger", "Sentry"]
FW = ["React", "Vue", "Angular", "Svelte", "Next.js", "Django", "FastAPI", "Rails",
      "Spring", "Laravel", "Gin", "Actix", "PyTorch", "TensorFlow", "JAX", "LangChain"]
PLACES = ["Berlin", "Tokyo", "Lisbon", "Austin", "Seattle", "Toronto", "Dublin",
          "Amsterdam", "Stockholm", "Munich", "Seoul", "Mumbai", "Sydney", "Tel Aviv",
          "Sao Paulo", "Mexico City", "Helsinki", "Zurich", "Prague", "Auckland"]
PRODUCTS = ["Phoenix", "Atlas", "Orion", "Nexus", "Aurora", "Apollo", "Vega", "Nova",
            "Zenith", "Cipher", "Forge", "Beacon", "Harbor", "Cedar", "Falcon", "Heron"]
ORGS = ["Acme Corp", "Globex", "Initech", "Umbra Labs", "Northwind", "Contoso",
        "Hooli", "Stark Industries", "Wayne Tech", "Cyberdyne"]
TOPICS = ["authentication", "caching", "deployment", "observability", "routing",
          "scheduling", "indexing", "migration", "rollout", "onboarding"]
UNITS = ["milliseconds", "seconds", "minutes", "gigabytes", "requests per second"]

# (template, qa_question, answer_slot_index)
# Each template uses {slot0} {slot1} ... and a qa_question with a matching {slotN}.
TEMPLATES = {
    "user_preferences": [
        ("The user prefers to utilize {s0} in order to accomplish {s1} tasks. However, they previously required additional tooling to facilitate the workflow. Nevertheless, they frequently investigate numerous alternatives and demonstrate considerable expertise.",
         "What language does the user prefer?", 0, LANGS),
        ("The user has established a preference for {s0} over alternatives. They additionally require strict typing and frequently communicate that maintainability is particularly important.",
         "What does the user prefer for typing?", 0, LANGS),
        ("The user prefers {s0} for interactive analysis and previously utilized {s1} for persistence. They frequently request additional examples and demonstrate considerable interest.",
         "What tool does the user prefer for analysis?", 0, FW),
    ],
    "project_context": [
        ("The project implements a microservices architecture utilizing {s0} containers. The team previously attempted to facilitate deployment through {s1} but subsequently determined that additional complexity was insufficiently advantageous.",
         "What container runtime does the project use?", 0, INFRA),
        ("The repository contains numerous modules relating to {s0}. Developers should investigate configuration files prior to modifying {s1} variables.",
         "What area do the modules relate to?", 0, TOPICS),
        ("The {s0} service utilizes {s1} for persistence and previously experienced performance issues that were subsequently resolved through query optimization.",
         "What database does the service use?", 1, DBS),
    ],
    "session_summary": [
        ("During the previous session, the assistant helped the user construct an endpoint for {s0}. The user requested additional documentation and consequently the assistant provided comprehensive examples utilizing {s1}.",
         "What framework was used in the session?", 1, FW),
        ("The assistant previously recognized that the user possesses significant experience with {s0}. Nevertheless, they required assistance regarding {s1} patterns.",
         "What does the user have experience with?", 0, FW),
        ("During the previous session the team utilized {s0} to facilitate the {s1} rollout. Subsequently they communicated that additional monitoring was required.",
         "What tool was used for the rollout?", 0, INFRA),
    ],
    "factual_memory": [
        ("The user's organization is located in {s0}. They frequently participate in conferences and communicate regularly with stakeholders regarding {s1} requirements.",
         "Where is the organization located?", 0, PLACES),
        ("The database utilizes {s0} version 15. The application previously experienced performance issues that were subsequently resolved through query optimization on the {s1} cluster.",
         "What database does the application use?", 0, DBS),
        ("The {s0} pipeline is monitored utilizing {s1}. The team previously encountered numerous incidents and subsequently established additional alerting rules.",
         "What tool monitors the pipeline?", 1, OBS),
    ],
    "instructions": [
        ("When responding, the assistant should communicate clearly and avoid excessively verbose explanations regarding {s0}. However, technical accuracy is particularly important and should never be compromised when utilizing {s1}.",
         "What should the assistant avoid regarding explanations?", 0, TOPICS),
        ("The user instructed the assistant to investigate {s0} errors thoroughly prior to proposing solutions. Additionally, they prefer concise summaries utilizing {s1} at the conclusion of each interaction.",
         "What kind of errors should be investigated?", 0, TOPICS),
        ("The assistant should always validate {s0} configuration prior to deployment and communicate any additional requirements to the {s1} team.",
         "What configuration should be validated?", 0, TOPICS),
    ],
    "mixed": [
        ("The user utilizes {s0} for note-taking and has established a Second Brain workflow. They previously requested that planning documents be stored in markdown format and demonstrate considerable interest in {s1} optimization.",
         "What tool does the user use for notes?", 0, FW),
        ("The {s0} project implements lexical normalization for agent memory and facilitates storage reduction while maintaining semantic equivalence utilizing {s1}.",
         "What project implements lexical normalization?", 0, PRODUCTS),
        ("The user works with {s0}, {s1}, and Kubernetes at Acme Corp and previously utilized GitHub Actions in order to facilitate CI/CD.",
         "What language does the user work with?", 0, LANGS),
    ],
    "high_verbosity": [
        ("The assistant should endeavor to accomplish {s0} tasks efficiently while simultaneously maintaining adequate communication. Consequently, it is particularly important to establish clear expectations at the commencement of each {s1} session.",
         "What kind of tasks should be accomplished?", 0, TOPICS),
        ("Numerous developers have previously attempted to implement {s0} utilizing {s1}. However, they frequently encountered considerable difficulties and subsequently terminated their efforts.",
         "What did developers attempt to implement?", 0, TOPICS),
        ("It is particularly important to facilitate {s0} prior to commencing the {s1} migration. Nevertheless, numerous stakeholders frequently request additional clarification.",
         "What should be facilitated prior to migration?", 0, TOPICS),
    ],
    "entity_heavy": [
        ("The user works with {s0}, {s1}, and Kubernetes at {s2}. They previously utilized GitHub Actions in order to facilitate CI/CD for the Phoenix microservice.",
         "At which organization does the user work?", 2, ORGS),
        ("The {s0} microservice utilizes {s1} for caching and {s2} for persistence. The team previously encountered numerous timeouts and subsequently optimized the connection pool.",
         "What does the microservice use for caching?", 1, INFRA),
        ("The {s0} pipeline deploys to {s1} clusters and is monitored by {s2}. The team previously required additional capacity and subsequently expanded the fleet.",
         "What does the pipeline deploy to?", 1, INFRA),
    ],
}

CATEGORY_PREFIX = {
    "user_preferences": "prefs", "project_context": "project", "session_summary": "session",
    "factual_memory": "facts", "instructions": "instr", "mixed": "mixed",
    "high_verbosity": "verbose", "entity_heavy": "entities",
}


def gen_passage(cat: str, idx: int) -> tuple[dict, dict]:
    tmpl, question, ans_idx, pool = random.choice(TEMPLATES[cat])
    slots_pools = [LANGS, DBS, INFRA, CI, OBS, FW, PLACES, PRODUCTS, ORGS, TOPICS, UNITS]
    # Fill {s0}..{sN} by scanning the template for slot markers.
    import re
    markers = sorted(set(re.findall(r"\{s(\d+)\}", tmpl)), key=lambda x: int(x))
    fills = {}
    for m in markers:
        n = int(m)
        # answer slot uses the template's pool; others pick from a rotating pool
        if n == ans_idx and pool:
            fills[f"s{n}"] = random.choice(pool)
        else:
            fills[f"s{n}"] = random.choice(slots_pools[n % len(slots_pools)])
    text = tmpl.format(**fills)
    answer = fills[f"s{ans_idx}"]
    pid = f"{CATEGORY_PREFIX[cat]}-{idx:03d}"
    keywords = [answer.lower()]
    sample = {"id": pid, "category": cat, "text": text}
    qa = {"qa_id": f"qa-{pid}", "question": question, "answer": answer,
          "keywords": keywords, "category": cat, "evidence_ids": [pid],
          "benchmark": "agent_corpus"}
    return sample, qa


def main() -> None:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    existing = data["samples"]
    existing_ids = {s["id"] for s in existing}
    samples = list(existing)
    qa_items = []

    # Hand-authored QA for the original 8 pinned items (keep stable).
    pinned = [
        ("qa-01", "What language does the user prefer?", "Python", ["python"], "preference", ["prefs-01"]),
        ("qa-02", "Where is the user's organization located?", "Berlin", ["berlin"], "factual", ["facts-01"]),
        ("qa-03", "What note-taking tool does the user use?", "Obsidian", ["obsidian"], "preference", ["mixed-01"]),
        ("qa-04", "Does the user prefer strict typing?", "TypeScript", ["typescript", "typing"], "preference", ["prefs-02"]),
        ("qa-05", "What architecture does the project use?", "microservices", ["microservices", "docker"], "project", ["project-01"]),
        ("qa-06", "What database does the application use?", "PostgreSQL", ["postgresql", "postgres"], "factual", ["facts-02"]),
        ("qa-07", "What does min-mem implement?", "lexical normalization", ["lexical", "normalization", "min-mem"], "project", ["mixed-02"]),
        ("qa-08", "What CI tool was previously used?", "GitHub Actions", ["github", "actions"], "project", ["entities-01"]),
    ]
    for qid, q, a, kw, cat, ev in pinned:
        qa_items.append({"qa_id": qid, "question": q, "answer": a, "keywords": kw,
                         "category": cat, "evidence_ids": ev, "benchmark": "agent_corpus"})

    target = 200
    per_cat = {c: 0 for c in TEMPLATES}
    # Count existing per category
    for s in existing:
        per_cat[s["category"]] = per_cat.get(s["category"], 0) + 1

    idx = 100  # new ids start at 100 to avoid colliding with existing -01/-02
    while len(samples) < target:
        # round-robin categories that are below quota
        cats = [c for c in TEMPLATES if per_cat[c] < (target // len(TEMPLATES)) + 2]
        if not cats:
            cats = list(TEMPLATES)
        cat = random.choice(cats)
        idx += 1
        sample, qa = gen_passage(cat, idx)
        if sample["id"] in existing_ids:
            continue
        samples.append(sample)
        qa_items.append(qa)
        per_cat[cat] += 1

    data["samples"] = samples[:target]
    data["description"] = (
        "Synthetic LLM agent memory corpus — formal verbose phrasing typical of "
        "persistent memory stores. 200 passages across eight agent-memory categories."
    )
    CORPUS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    QA_OUT.write_text(json.dumps({"qa": qa_items[:target]}, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {len(data['samples'])} samples, {len(qa_items[:target])} qa items")
    from collections import Counter
    print("categories:", Counter(s["category"] for s in data["samples"]))


if __name__ == "__main__":
    main()
