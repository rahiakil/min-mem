from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from min_mem import MinMemConverter
from min_mem.bootstrap import doctor_report, init_project
from min_mem.dictionary import MinDictionary, resolve_dict_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="min-mem",
        description="Minify memory text using a minimal synonym dictionary (nouns preserved).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Install dictionary, NLTK data, and optional Cursor hook")
    init_cmd.add_argument(
        "--no-cursor",
        action="store_true",
        help="Skip installing .cursor/hooks.json session hook",
    )
    init_cmd.add_argument("--force", action="store_true", help="Overwrite user dictionary")

    doctor = sub.add_parser("doctor", help="Verify install, dictionary, and NLTK data")
    doctor.add_argument("--json", action="store_true", help="JSON output")

    minify = sub.add_parser("minify", help="Reduce text size while preserving meaning")
    minify.add_argument("text", nargs="?", help="Text to minify (or use --file)")
    minify.add_argument("-f", "--file", type=Path, help="Read text from a file")
    minify.add_argument("-d", "--dict", type=Path, help="Custom dictionary JSON path")
    minify.add_argument("--json", action="store_true", help="Emit structured JSON output")
    minify.add_argument("-v", "--verbose", action="store_true", help="Show replacements")

    dict_cmd = sub.add_parser("dict", help="Inspect the minimal dictionary")
    dict_cmd.add_argument("-d", "--dict", type=Path, help="Custom dictionary JSON path")
    dict_cmd.add_argument("--count", action="store_true", help="Print entry count only")

    stats = sub.add_parser("stats", help="Show compression stats for text")
    stats.add_argument("text", nargs="?", help="Text to analyze (or use --file)")
    stats.add_argument("-f", "--file", type=Path, help="Read text from a file")
    stats.add_argument("-d", "--dict", type=Path, help="Custom dictionary JSON path")

    measure = sub.add_parser("measure", help="Benchmark minification on a file or stdin")
    measure.add_argument("-f", "--file", type=Path, help="Memory file to measure")
    measure.add_argument("-d", "--dict", type=Path, help="Custom dictionary JSON path")
    measure.add_argument("--json", action="store_true", help="JSON output")

    return parser


def _read_input(text: str | None, file: Path | None) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is not None:
        return text
    return sys.stdin.read()


def _converter(dict_path: Path | None) -> MinMemConverter:
    if dict_path:
        return MinMemConverter.from_dict_path(dict_path)
    return MinMemConverter()


def cmd_init(args: argparse.Namespace) -> int:
    result = init_project(cursor_hook=not args.no_cursor, force_dict=args.force)
    print("Min-Mem initialized.")
    print(f"  Dictionary: {result['dictionary']} ({result['entries']} entries)")
    if result.get("cursor_hook"):
        print(f"  Cursor hook: {result['cursor_hook']}")
    print(f"  Metrics:     {result['metrics_file']}")
    print("\nRun `min-mem doctor` to verify.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "OK" if report["ok"] else "ISSUES"
        print(f"min-mem {report['version']} — {status}")
        print(f"  Dictionary: {report['dictionary_entries']} entries ({report['dictionary_path']})")
        print(f"  User dict:  {'installed' if report['user_dict_installed'] else 'using bundled'}")
        print(f"  NLTK:       {'ready' if report['nltk_ok'] else report['nltk_error']}")
        if report.get("metrics"):
            m = report["metrics"]
            print(f"  Lifetime:   {m.get('total_chars_saved', 0)} chars saved across sessions")
    return 0 if report["ok"] else 1


def cmd_minify(args: argparse.Namespace) -> int:
    content = _read_input(args.text, args.file)
    converter = _converter(args.dict)
    result = converter.minify(content)

    if args.json:
        payload = {
            "original": result.original,
            "minified": result.minified,
            "original_chars": result.original_chars,
            "minified_chars": result.minified_chars,
            "chars_saved": result.chars_saved,
            "savings_ratio": round(result.savings_ratio, 4),
            "replacements": [
                {"original": r.original, "replacement": r.replacement, "position": r.position}
                for r in result.replacements
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.minified)
        if args.verbose and result.replacements:
            print("\n--- replacements ---", file=sys.stderr)
            for r in result.replacements:
                print(f"  {r.original!r} -> {r.replacement!r}", file=sys.stderr)
            print(
                f"\nSaved {result.chars_saved} chars "
                f"({result.savings_ratio:.1%} of {result.original_chars})",
                file=sys.stderr,
            )
    return 0


def cmd_dict(args: argparse.Namespace) -> int:
    path = args.dict or resolve_dict_path()
    dictionary = MinDictionary.from_path(path)
    if args.count:
        print(len(dictionary))
        return 0

    entries = sorted(
        ((e.source, e.target) for e in dictionary.word_entries()),
        key=lambda pair: pair[0],
    )
    phrases = sorted(
        ((e.source, e.target) for e in dictionary.phrase_entries()),
        key=lambda pair: pair[0],
    )
    print(f"# minimal dictionary ({len(dictionary)} entries)\n")
    print("## phrases")
    for source, target in phrases:
        print(f"  {source!r} -> {target!r}")
    print("\n## words")
    for source, target in entries:
        print(f"  {source!r} -> {target!r}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    content = _read_input(args.text, args.file)
    converter = _converter(args.dict)
    result = converter.minify(content)
    print(f"original:  {result.original_chars} chars")
    print(f"minified:  {result.minified_chars} chars")
    print(f"saved:     {result.chars_saved} chars ({result.savings_ratio:.1%})")
    print(f"swaps:     {len(result.replacements)}")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    content = _read_input(None, args.file)
    converter = _converter(args.dict)
    result = converter.minify(content)
    payload = {
        "original_chars": result.original_chars,
        "minified_chars": result.minified_chars,
        "chars_saved": result.chars_saved,
        "savings_pct": round(result.savings_ratio * 100, 2),
        "replacements": len(result.replacements),
        "dictionary": str(resolve_dict_path(args.dict)),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"chars:  {result.original_chars} → {result.minified_chars}  (−{result.chars_saved}, {payload['savings_pct']}%)")
        print(f"swaps:  {len(result.replacements)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "init": cmd_init,
        "doctor": cmd_doctor,
        "minify": cmd_minify,
        "dict": cmd_dict,
        "stats": cmd_stats,
        "measure": cmd_measure,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
