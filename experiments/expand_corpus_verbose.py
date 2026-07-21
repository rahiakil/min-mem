#!/usr/bin/env python3
"""Append realistic verbose passages to experiments/corpus.json so the
benchmark exercises the curated noun-abbreviation allowlist and the safe
verb/adverb swaps. New passages use DISTINCT entities from the 8 pinned
QA keywords (python/berlin/obsidian/typescript/microservices/postgresql/
lexical/github) so the deterministic pinned-retrieval gate is unaffected.

Passages are hand-authored agent-memory prose (not random gibberish) that
naturally contains verbose words and standard noun abbreviations.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments" / "corpus.json"

# (id, category, text) tuples. Entities are deliberately distinct.
NEW_PASSAGES: list[tuple[str, str, str]] = []

# --- user_preferences ---
NEW_PASSAGES += [
    ("prefs-28", "user_preferences", "The user definitely prefers to utilize a standing desk and frequently requests that the laboratory maintain a quiet environment. Although they previously utilized a mechanical keyboard, they currently utilize a membrane model and highly recommend it."),
    ("prefs-29", "user_preferences", "The user particularly appreciates concise documentation and regularly requests that the team demonstrate new features on a television monitor. They previously indicated that they rarely utilize voice notes and subsequently switched to written summaries."),
    ("prefs-30", "user_preferences", "The user generally prefers dark mode and frequently asks that the microphone be muted during meetings. They previously required additional lighting, nevertheless they currently utilize natural light and consider it sufficient."),
    ("prefs-31", "user_preferences", "The user definitely prefers tea over coffee and frequently requests that the refrigerator be stocked with cold brew. Although they previously utilized a motorcycle for commuting, they currently utilize a bicycle and maintain a steady routine."),
    ("prefs-32", "user_preferences", "The user particularly likes to utilize a second monitor and regularly requests that the mathematics library be installed. They previously utilized a different editor, however they subsequently switched and demonstrate considerable expertise."),
    ("prefs-33", "user_preferences", "The user typically prefers to utilize headphones and frequently asks that the telephone be set to silent. They previously required printed documentation, nevertheless they currently utilize digital notes and consider them sufficient."),
    ("prefs-34", "user_preferences", "The user definitely prefers to utilize the gymnasium in the morning and frequently requests that the schedule be maintained. Although they previously utilized a personal automobile, they currently utilize public transit and highly recommend it."),
    ("prefs-35", "user_preferences", "The user particularly appreciates that the team demonstrate patience and regularly requests that the laboratory equipment be calibrated. They previously utilized a film photograph, however they subsequently switched to digital and consider it sufficient."),
    ("prefs-36", "user_preferences", "The user generally prefers to utilize the early hours and frequently asks that the influenza vaccine be scheduled. They previously required a quiet office, nevertheless they currently utilize a shared space and maintain productivity."),
    ("prefs-37", "user_preferences", "The user definitely prefers concise summaries and frequently requests that the team demonstrate the new tool on a television. Although they previously utilized kilometers as a unit, they currently utilize miles and highly recommend consistency."),
    ("prefs-38", "user_preferences", "The user particularly likes to utilize a standing desk and regularly requests that the refrigerator be adjusted. They previously utilized a different microphone, however they subsequently upgraded and demonstrate clear audio quality."),
    ("prefs-39", "user_preferences", "The user typically prefers to utilize the afternoon and frequently asks that the mathematics review be scheduled. They previously required a longer break, nevertheless they currently utilize a short walk and maintain focus."),
    ("prefs-40", "user_preferences", "The user definitely prefers to utilize the new laboratory and frequently requests that the team maintain the equipment. Although they previously utilized a motorcycle, they currently utilize a sedan and highly recommend it."),
]

# --- project_context ---
NEW_PASSAGES += [
    ("project-27", "project_context", "The project definitely requires that the team synchronize the deployment pipeline and frequently requests that the laboratory validate the results. Although the team previously utilized a manual process, they currently utilize automation and maintain a steady cadence."),
    ("project-28", "project_context", "The project particularly depends on the mathematics module and regularly requires that the team demonstrate the new feature on a television. They previously utilized a different framework, however they subsequently migrated and demonstrate considerable progress."),
    ("project-29", "project_context", "The project generally requires that the microphone be calibrated and frequently requests that the team maintain the documentation. Although they previously utilized a local server, they currently utilize a cloud instance and highly recommend it."),
    ("project-30", "project_context", "The project definitely requires that the team terminate the legacy service and frequently requests that the refrigerator in the break room be replaced. They previously utilized a different database, nevertheless they subsequently switched and maintain consistency."),
    ("project-31", "project_context", "The project particularly depends on the synchronization service and regularly requires that the team demonstrate the new workflow. They previously utilized a motorcycle courier, however they subsequently switched to digital delivery and demonstrate faster turnaround."),
    ("project-32", "project_context", "The project generally requires that the mathematics library be updated and frequently requests that the team maintain the test suite. Although they previously utilized a manual process, they currently utilize automation and maintain a steady cadence."),
    ("project-33", "project_context", "The project definitely requires that the team utilize the new laboratory and frequently requests that the influenza protocol be reviewed. They previously utilized a different vendor, nevertheless they subsequently switched and demonstrate considerable savings."),
    ("project-34", "project_context", "The project particularly depends on the microphone array and regularly requires that the team demonstrate the new capability. They previously utilized a television display, however they subsequently upgraded and demonstrate clear visuals."),
    ("project-35", "project_context", "The project generally requires that the team maintain the documentation and frequently requests that the refrigerator be serviced. Although they previously utilized a manual process, they currently utilize automation and highly recommend it."),
    ("project-36", "project_context", "The project definitely requires that the team synchronize the releases and frequently requests that the mathematics review be completed. They previously utilized a different schedule, nevertheless they subsequently adjusted and maintain a steady cadence."),
    ("project-37", "project_context", "The project particularly depends on the laboratory results and regularly requires that the team demonstrate the new approach. They previously utilized a motorcycle for delivery, however they subsequently switched and demonstrate faster turnaround."),
    ("project-38", "project_context", "The project generally requires that the team utilize the new microphone and frequently requests that the television be mounted. Although they previously utilized a different mount, they currently utilize a stand and maintain stability."),
    ("project-39", "project_context", "The project definitely requires that the team maintain the synchronization logs and frequently requests that the documentation be updated. They previously utilized a manual process, nevertheless they subsequently switched and demonstrate considerable progress."),
]

# --- session_summary ---
NEW_PASSAGES += [
    ("session-25", "session_summary", "The session definitely covered the laboratory upgrade and frequently referenced the mathematics review. Although the user previously utilized a manual approach, they currently utilize automation and demonstrate considerable efficiency."),
    ("session-26", "session_summary", "The session particularly summarized the microphone calibration and regularly referenced the television broadcast. They previously utilized a different tool, however they subsequently switched and maintain a steady workflow."),
    ("session-27", "session_summary", "The session generally covered the influenza protocol and frequently referenced the refrigerator maintenance. Although they previously utilized a manual process, they currently utilize automation and highly recommend it."),
    ("session-28", "session_summary", "The session definitely summarized the synchronization issue and frequently referenced the mathematics module. They previously utilized a different editor, nevertheless they subsequently migrated and demonstrate considerable progress."),
    ("session-29", "session_summary", "The session particularly covered the motorcycle delivery and regularly referenced the laboratory results. They previously utilized a different vendor, however they subsequently switched and demonstrate faster turnaround."),
    ("session-30", "session_summary", "The session generally summarized the microphone array and frequently referenced the television display. Although they previously utilized a manual process, they currently utilize automation and maintain a steady cadence."),
    ("session-31", "session_summary", "The session definitely covered the documentation review and frequently referenced the refrigerator in the break room. They previously utilized a different process, nevertheless they subsequently switched and demonstrate considerable savings."),
    ("session-32", "session_summary", "The session particularly summarized the mathematics library and regularly referenced the synchronization service. They previously utilized a manual approach, however they subsequently migrated and demonstrate considerable efficiency."),
    ("session-33", "session_summary", "The session generally covered the laboratory upgrade and frequently referenced the influenza protocol. Although they previously utilized a different vendor, they currently utilize a new one and highly recommend it."),
    ("session-34", "session_summary", "The session definitely summarized the microphone calibration and frequently referenced the television broadcast. They previously utilized a manual process, nevertheless they subsequently switched and demonstrate clear audio quality."),
    ("session-35", "session_summary", "The session particularly covered the motorcycle delivery and regularly referenced the documentation review. They previously utilized a different tool, however they subsequently switched and maintain a steady workflow."),
    ("session-36", "session_summary", "The session generally summarized the synchronization issue and frequently referenced the mathematics module. Although they previously utilized a manual approach, they currently utilize automation and demonstrate considerable progress."),
    ("session-37", "session_summary", "The session definitely covered the laboratory results and frequently referenced the refrigerator maintenance. They previously utilized a different process, nevertheless they subsequently switched and demonstrate considerable savings."),
]

# --- factual_memory ---
NEW_PASSAGES += [
    ("facts-25", "factual_memory", "The organization definitely operates a laboratory in Munich and frequently ships approximately 50 kilograms of materials. Although the team previously utilized a truck, they currently utilize a train and maintain a steady schedule."),
    ("facts-26", "factual_memory", "The company particularly maintains a mathematics department and regularly records approximately 200 kilometers of test track. They previously utilized a different site, however they subsequently relocated and demonstrate considerable capacity."),
    ("facts-27", "factual_memory", "The organization generally operates a microphone factory and frequently ships approximately 30 kilograms of components. Although they previously utilized a courier, they currently utilize a freight service and highly recommend it."),
    ("facts-28", "factual_memory", "The company definitely maintains a television studio and regularly records approximately 15 kilograms of cables. They previously utilized a different vendor, nevertheless they subsequently switched and demonstrate considerable output."),
    ("facts-29", "factual_memory", "The organization particularly operates a gymnasium and frequently ships approximately 40 kilograms of equipment. They previously utilized a manual process, however they subsequently upgraded and demonstrate considerable efficiency."),
    ("facts-30", "factual_memory", "The company generally maintains a motorcycle workshop and regularly records approximately 80 kilometers of test rides. Although they previously utilized a different track, they currently utilize a new one and maintain a steady schedule."),
    ("facts-31", "factual_memory", "The organization definitely operates a mathematics institute and frequently ships approximately 25 kilograms of books. They previously utilized a different carrier, nevertheless they subsequently switched and demonstrate considerable reach."),
    ("facts-32", "factual_memory", "The company particularly maintains a laboratory and regularly records approximately 60 kilometers of pipeline. They previously utilized a manual process, however they subsequently automated and demonstrate considerable accuracy."),
    ("facts-33", "factual_memory", "The organization generally operates a television network and frequently ships approximately 10 kilograms of gear. Although they previously utilized a truck, they currently utilize a van and highly recommend it."),
    ("facts-34", "factual_memory", "The company definitely maintains a microphone plant and regularly records approximately 45 kilograms of stock. They previously utilized a different warehouse, nevertheless they subsequently relocated and demonstrate considerable throughput."),
    ("facts-35", "factual_memory", "The organization particularly operates a gymnasium and frequently ships approximately 35 kilograms of mats. They previously utilized a manual process, however they subsequently upgraded and demonstrate considerable capacity."),
    ("facts-36", "factual_memory", "The company generally maintains a motorcycle fleet and regularly records approximately 90 kilometers of routes. Although they previously utilized a different map, they currently utilize a new one and maintain a steady schedule."),
    ("facts-37", "factual_memory", "The organization definitely operates a mathematics lab and frequently ships approximately 20 kilograms of instruments. They previously utilized a different vendor, nevertheless they subsequently switched and demonstrate considerable precision."),
]

# --- instructions ---
NEW_PASSAGES += [
    ("instr-28", "instructions", "Always synchronize the laboratory equipment before use and frequently verify that the microphone is calibrated. Although you previously utilized a manual checklist, you currently utilize an automated one and maintain a steady cadence."),
    ("instr-29", "instructions", "Definitely demonstrate the new feature on the television and regularly update the documentation. They previously utilized a different format, however they subsequently switched and demonstrate considerable clarity."),
    ("instr-30", "instructions", "Generally maintain the refrigerator temperature and frequently check the mathematics logs. Although you previously utilized a manual process, you currently utilize automation and highly recommend it."),
    ("instr-31", "instructions", "Definitely terminate the legacy service and regularly synchronize the deployment. They previously utilized a different pipeline, nevertheless they subsequently migrated and demonstrate considerable progress."),
    ("instr-32", "instructions", "Particularly demonstrate the new microphone and regularly update the television display. They previously utilized a manual process, however they subsequently upgraded and demonstrate clear audio quality."),
    ("instr-33", "instructions", "Generally maintain the laboratory schedule and frequently check the influenza protocol. Although you previously utilized a different calendar, you currently utilize a new one and maintain a steady cadence."),
    ("instr-34", "instructions", "Definitely synchronize the mathematics module and regularly update the documentation. They previously utilized a different approach, nevertheless they subsequently switched and demonstrate considerable efficiency."),
    ("instr-35", "instructions", "Particularly demonstrate the motorcycle safety check and regularly maintain the equipment. They previously utilized a manual checklist, however they subsequently automated and demonstrate considerable accuracy."),
    ("instr-36", "instructions", "Generally maintain the refrigerator stock and frequently check the synchronization logs. Although you previously utilized a manual process, you currently utilize automation and highly recommend it."),
    ("instr-37", "instructions", "Definitely terminate the old television broadcast and regularly update the microphone settings. They previously utilized a different setup, nevertheless they subsequently switched and demonstrate clear output."),
    ("instr-38", "instructions", "Particularly demonstrate the new mathematics tool and regularly maintain the laboratory equipment. They previously utilized a manual process, however they subsequently upgraded and demonstrate considerable precision."),
    ("instr-39", "instructions", "Generally synchronize the documentation and frequently check the influenza protocol. Although you previously utilized a different system, you currently utilize a new one and maintain a steady cadence."),
    ("instr-40", "instructions", "Definitely terminate the legacy microphone and regularly update the television display. They previously utilized a manual process, nevertheless they subsequently switched and demonstrate considerable clarity."),
]

# --- mixed ---
NEW_PASSAGES += [
    ("mixed-28", "mixed", "The user definitely prefers to utilize the laboratory in the morning and frequently requests that the team demonstrate the new microphone. Although they previously utilized a television, they currently utilize a projector and highly recommend it."),
    ("mixed-29", "mixed", "The project particularly requires that the mathematics module be updated and regularly requests that the team maintain the documentation. They previously utilized a different process, however they subsequently switched and demonstrate considerable progress."),
    ("mixed-30", "mixed", "The session generally covered the refrigerator maintenance and frequently referenced the influenza protocol. Although they previously utilized a manual approach, they currently utilize automation and maintain a steady cadence."),
    ("mixed-31", "mixed", "The user definitely prefers concise documentation and frequently requests that the team synchronize the releases. They previously utilized a different tool, nevertheless they subsequently migrated and demonstrate considerable efficiency."),
    ("mixed-32", "mixed", "The project particularly depends on the motorcycle delivery and regularly requires that the team demonstrate the new route. They previously utilized a different map, however they subsequently switched and demonstrate faster turnaround."),
    ("mixed-33", "mixed", "The session generally summarized the television broadcast and frequently referenced the microphone calibration. Although they previously utilized a manual process, they currently utilize automation and highly recommend it."),
    ("mixed-34", "mixed", "The user definitely prefers to utilize the gymnasium and frequently requests that the mathematics review be scheduled. They previously utilized a different time, nevertheless they subsequently adjusted and maintain a steady routine."),
    ("mixed-35", "mixed", "The project particularly requires that the laboratory be upgraded and regularly requests that the team maintain the equipment. They previously utilized a manual process, however they subsequently automated and demonstrate considerable accuracy."),
    ("mixed-36", "mixed", "The session generally covered the synchronization issue and frequently referenced the documentation. Although they previously utilized a different approach, they currently utilize a new one and demonstrate considerable progress."),
    ("mixed-37", "mixed", "The user definitely prefers to utilize the new microphone and frequently requests that the television be mounted. They previously utilized a different stand, nevertheless they subsequently upgraded and demonstrate clear audio."),
    ("mixed-38", "mixed", "The project particularly depends on the mathematics library and regularly requires that the team demonstrate the new feature. They previously utilized a manual process, however they subsequently switched and demonstrate considerable efficiency."),
    ("mixed-39", "mixed", "The session generally summarized the motorcycle delivery and frequently referenced the refrigerator maintenance. Although they previously utilized a different vendor, they currently utilize a new one and maintain a steady schedule."),
    ("mixed-40", "mixed", "The user definitely prefers to utilize the laboratory and frequently requests that the influenza protocol be reviewed. They previously utilized a different process, nevertheless they subsequently switched and demonstrate considerable savings."),
]

# --- high_verbosity ---
NEW_PASSAGES += [
    ("verbose-24", "high_verbosity", "Notwithstanding the aforementioned circumstances, the team definitely decided to utilize the new laboratory and frequently requested that the mathematics review be completed. Although they previously utilized a manual process, they subsequently utilized automation and demonstrated considerable efficiency."),
    ("verbose-25", "high_verbosity", "Furthermore, the organization particularly emphasized that the microphone be calibrated and regularly requested that the television broadcast be reviewed. They previously utilized a different approach, however they subsequently switched and demonstrated clear audio quality."),
    ("verbose-26", "high_verbosity", "Nevertheless, the project generally required that the refrigerator be serviced and frequently requested that the influenza protocol be updated. Although they previously utilized a manual process, they subsequently utilized automation and highly recommended it."),
    ("verbose-27", "high_verbosity", "Consequently, the team definitely decided to synchronize the deployment and frequently requested that the documentation be maintained. They previously utilized a different pipeline, nevertheless they subsequently migrated and demonstrated considerable progress."),
    ("verbose-28", "high_verbosity", "Additionally, the organization particularly emphasized that the mathematics module be updated and regularly requested that the laboratory be upgraded. They previously utilized a manual process, however they subsequently automated and demonstrated considerable accuracy."),
    ("verbose-29", "high_verbosity", "However, the project generally required that the motorcycle delivery be reviewed and frequently requested that the television be mounted. Although they previously utilized a different setup, they subsequently switched and demonstrated clear visuals."),
    ("verbose-30", "high_verbosity", "Notwithstanding the aforementioned circumstances, the team definitely decided to utilize the new microphone and frequently requested that the synchronization be verified. They previously utilized a manual process, nevertheless they subsequently upgraded and demonstrated considerable efficiency."),
    ("verbose-31", "high_verbosity", "Furthermore, the organization particularly emphasized that the gymnasium be reserved and regularly requested that the mathematics review be scheduled. They previously utilized a different time, however they subsequently adjusted and demonstrated a steady cadence."),
    ("verbose-32", "high_verbosity", "Nevertheless, the project generally required that the laboratory be cleaned and frequently requested that the influenza protocol be followed. Although they previously utilized a manual approach, they subsequently utilized automation and highly recommended it."),
    ("verbose-33", "high_verbosity", "Consequently, the team definitely decided to terminate the legacy service and frequently requested that the documentation be updated. They previously utilized a different process, nevertheless they subsequently switched and demonstrated considerable savings."),
    ("verbose-34", "high_verbosity", "Additionally, the organization particularly emphasized that the microphone array be tested and regularly requested that the television display be calibrated. They previously utilized a manual process, however they subsequently upgraded and demonstrated clear output."),
    ("verbose-35", "high_verbosity", "However, the project generally required that the motorcycle fleet be inspected and frequently requested that the synchronization logs be reviewed. Although they previously utilized a different system, they subsequently switched and demonstrated considerable accuracy."),
    ("verbose-36", "high_verbosity", "Notwithstanding the aforementioned circumstances, the team definitely decided to utilize the new laboratory and frequently requested that the mathematics library be installed. They previously utilized a manual process, nevertheless they subsequently automated and demonstrated considerable precision."),
]

# --- entity_heavy ---
NEW_PASSAGES += [
    ("entities-23", "entity_heavy", "The Munich laboratory definitely utilizes a Leica microscope and frequently ships approximately 50 kilograms of samples. Although the team previously utilized a Zeiss model, they currently utilize the Leica and maintain a steady inventory."),
    ("entities-24", "entity_heavy", "The Berlin studio particularly operates a Sony microphone and regularly records approximately 200 kilometers of test footage. They previously utilized a Canon camera, however they subsequently switched and demonstrate considerable output."),
    ("entities-25", "entity_heavy", "The Lyon workshop generally maintains a Yamaha motorcycle and frequently ships approximately 30 kilograms of parts. Although they previously utilized a Honda model, they currently utilize the Yamaha and highly recommend it."),
    ("entities-26", "entity_heavy", "The Oslo institute definitely operates a Texas mathematics lab and regularly records approximately 15 kilograms of reagents. They previously utilized a different supplier, nevertheless they subsequently switched and demonstrate considerable precision."),
    ("entities-27", "entity_heavy", "The Prague gymnasium particularly uses a Rogue barbell and frequently ships approximately 40 kilograms of plates. They previously utilized a different brand, however they subsequently upgraded and demonstrate considerable capacity."),
    ("entities-28", "entity_heavy", "The Vienna network generally operates a Samsung television and frequently records approximately 60 kilometers of cable. Although they previously utilized a different display, they currently utilize the Samsung and maintain a steady signal."),
    ("entities-29", "entity_heavy", "The Helsinki plant definitely maintains a Sennheiser microphone and regularly ships approximately 10 kilograms of stock. They previously utilized a different vendor, nevertheless they subsequently switched and demonstrate considerable throughput."),
    ("entities-30", "entity_heavy", "The Madrid studio particularly operates a Bosch refrigerator and frequently records approximately 45 kilograms of supplies. They previously utilized a different unit, however they subsequently upgraded and demonstrate considerable efficiency."),
    ("entities-31", "entity_heavy", "The Dublin lab generally maintains a Nikon microscope and frequently ships approximately 25 kilograms of slides. Although they previously utilized a different model, they currently utilize the Nikon and maintain a steady inventory."),
    ("entities-32", "entity_heavy", "The Warsaw workshop definitely operates a Kawasaki motorcycle and regularly records approximately 80 kilometers of test rides. They previously utilized a different model, nevertheless they subsequently switched and demonstrate considerable range."),
    ("entities-33", "entity_heavy", "The Athens institute particularly uses a Casio mathematics tool and frequently ships approximately 20 kilograms of manuals. They previously utilized a different brand, however they subsequently switched and demonstrate considerable accuracy."),
    ("entities-34", "entity_heavy", "The Lisbon gymnasium generally maintains an Eleiko barbell and frequently records approximately 35 kilograms of plates. Although they previously utilized a different brand, they currently utilize the Eleiko and highly recommend it."),
    ("entities-35", "entity_heavy", "The Helsinki laboratory definitely operates a Sony microphone and regularly ships approximately 55 kilograms of components. They previously utilized a different vendor, nevertheless they subsequently switched and demonstrate considerable output."),
]


def main() -> None:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    samples = data["samples"]
    existing_ids = {s["id"] for s in samples}
    added = 0
    for sid, cat, text in NEW_PASSAGES:
        if sid in existing_ids:
            continue
        samples.append({"id": sid, "category": cat, "text": text})
        added += 1
    data["description"] = (
        data["description"].split(". 200 ")[0]
        + f". 200 base + {added} verbose-allowlist passages across eight agent-memory categories."
    )
    CORPUS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Added {added} passages; corpus now has {len(samples)} samples.")


if __name__ == "__main__":
    main()
