"""CLI: python -m arch_cost {registry|check|run|report}"""

import argparse

from .registry import MODELS, print_registry


def main() -> None:
    ap = argparse.ArgumentParser(prog="arch_cost")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("registry", help="show the model/architecture registry")
    sub.add_parser("check", help="verify registry IDs against live OpenRouter catalog")

    runp = sub.add_parser("run", help="execute the experiment")
    runp.add_argument("--models", nargs="*", default=None,
                      help="subset of registry IDs (default: all available)")
    runp.add_argument("--suites", nargs="*", default=None,
                      choices=["easy", "reasoning", "tool"])
    runp.add_argument("--repeats", type=int, default=3)
    runp.add_argument("--out", default="results.jsonl")
    runp.add_argument("--seed", type=int, default=42)

    repp = sub.add_parser("report", help="analyze results")
    repp.add_argument("--results", default="results.jsonl")

    args = ap.parse_args()

    if args.cmd == "registry":
        print_registry()
        return

    if args.cmd in ("check", "run"):
        from .pricing import fetch_catalog, resolve
        catalog = fetch_catalog()
        ids = args.models if getattr(args, "models", None) else [m.id for m in MODELS]
        available, missing = resolve(ids, catalog)
        for m in missing:
            print(f"  !! NOT on OpenRouter right now: {m}")
        for m in available:
            p = catalog[m]
            print(f"  ok {m:46} ${p['prompt']*1e6:.3f}/M in  "
                  f"${p['completion']*1e6:.3f}/M out")
        if args.cmd == "run":
            from .runner import run_experiment
            run_experiment(available, catalog, repeats=args.repeats,
                           suites=args.suites, seed=args.seed,
                           out_path=args.out)
        return

    if args.cmd == "report":
        from .analyze import report
        report(args.results)


if __name__ == "__main__":
    main()
