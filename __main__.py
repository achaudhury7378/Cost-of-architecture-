from analyze import report
from registry import MODELS
from runner import run_experiment
from tasks import EASY, TOOL

REPEATS = 1
SEED = 42
SUITES = list({"easy" : EASY, "tool" : TOOL})

def main() -> None:
    # if args.cmd in ("check", "run"):
    from pricing import fetch_catalog, resolve
    catalog = fetch_catalog()
    ids = [m.id for m in MODELS]
    available, missing = resolve(ids, catalog)
    for m in missing:
        print(f"  !! NOT on OpenRouter right now: {m}")
    for m in available:
        p = catalog[m]
        print(f"  ok {m:46} ${p['prompt']*1e6:.3f}/M in  "
                f"${p['completion']*1e6:.3f}/M out")
    
    out_path = run_experiment(available, catalog, repeats=REPEATS,suites=SUITES, seed=SEED,out_path="./results.jsonl")
    report(out_path)
    


if __name__ == "__main__":
    main()
    # report("./results.jsonl")
