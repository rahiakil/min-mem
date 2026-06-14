# Agent instructions — min-mem

## Second Brain (required)

**All planning, research notes, thinking processes, and session logs must be written to Obsidian:**

```
~/obsidian2nd/Second Brain/
```

### When to write there

- Starting or refining a plan
- Exploring design options or trade-offs
- Research findings and literature-style notes
- Session summaries after meaningful work
- Roadmaps, hypotheses, open questions
- Anything that would otherwise live only in chat

### When working on min-mem specifically

Add notes under:

```
~/obsidian2nd/Second Brain/min-mem/
```

Start from the hub: `min-mem - index.md`

### File conventions

| Type | Pattern | Example |
|------|---------|---------|
| Project hub | `{project} - index.md` | `min-mem - index.md` |
| Concept | `{Topic} - Concept.md` | `Memory Minification - Concept.md` |
| Implementation | `{Topic} - Implementation.md` | |
| Planning | `Planning - {topic}.md` | `Planning - semantic validation.md` |
| Session log | `Session YYYY-MM-DD - {topic}.md` | `Session 2026-06-12 - min-mem build.md` |

### Format

- Markdown only (`.md`)
- Use Obsidian `[[wikilinks]]` between related notes
- Include tags at bottom: `#project/min-mem`, topic tags
- Be substantive — capture reasoning, not just conclusions

### Do not

- Put long-term thinking only in chat or ephemeral comments
- Create planning docs in the repo unless they are operational (README, AGENTS.md, code docs)
- Skip the Second Brain when the user asks for planning or research

## Project context

**min-mem** minifies LLM memory text via a minimal synonym dictionary. Nouns are never replaced. See `README.md` and Second Brain cluster [[min-mem - index]] at `~/obsidian2nd/Second Brain/min-mem/`.

## Code principles

- Minimal diff; match existing style
- Extend `min_dict.json` for new synonyms — avoid hardcoding words in Python
- Preserve noun POS-gating in any converter changes
- Run `pytest` after converter/dictionary changes

## Repo layout

```
min_dict.json           # minimal synonym dictionary (primary research artifact)
src/min_mem/            # converter, dictionary loader, CLI
tests/
~/obsidian2nd/Second Brain/min-mem/   # thinking & planning (not in git)
```
