#!/usr/bin/env python3
"""
Moulinette de la Piscine IA.

Usage :
    python3 moulinette/moulinette.py module00          # corrige tout le module
    python3 moulinette/moulinette.py module00 ex03     # corrige un seul exercice
    python3 moulinette/moulinette.py module06 --long   # inclut les entraînements longs

Comme à la piscine : la correction s'arrête au premier KO.
"""

import argparse
import importlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDU = os.environ.get("IA_RENDU", os.path.join(ROOT, "rendu"))
TESTS_DIR = os.path.join(ROOT, "moulinette", "tests")

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_tests(module_name):
    sys.path.insert(0, TESTS_DIR)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"{RED}Module inconnu : {module_name}{RESET}")
        sys.exit(1)


def run_single(module_name, ex_name):
    """Exécuté dans un sous-processus : lance le test d'un exercice."""
    tests = load_tests(module_name)
    ex_dir = os.path.join(RENDU, module_name, ex_name)
    test_fn = dict((name, fn) for name, fn, *_ in tests.EXERCISES)[ex_name]
    if not os.path.isdir(ex_dir):
        print(f"dossier introuvable : rendu/{module_name}/{ex_name}/")
        sys.exit(1)
    sys.path.insert(0, ex_dir)
    os.chdir(ex_dir)
    try:
        test_fn(ex_dir)
    except AssertionError as e:
        print(str(e) or "assertion échouée")
        sys.exit(1)
    except Exception as e:
        import traceback

        tb = traceback.extract_tb(e.__traceback__)
        student = [f for f in tb if f.filename.startswith(ex_dir)]
        where = f" ({os.path.basename(student[-1].filename)}:{student[-1].lineno})" if student else ""
        print(f"ton code a planté : {type(e).__name__}: {e}{where}")
        sys.exit(1)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Moulinette de la Piscine IA")
    parser.add_argument("module", help="ex: module00")
    parser.add_argument("exercise", nargs="?", help="ex: ex02 (optionnel)")
    parser.add_argument("--long", action="store_true", help="inclut les exercices d'entraînement longs")
    parser.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._run:
        run_single(args.module, args.exercise)
        return

    tests = load_tests(args.module)
    exercises = tests.EXERCISES
    if args.exercise:
        exercises = [e for e in exercises if e[0] == args.exercise]
        if not exercises:
            print(f"{RED}Exercice inconnu : {args.exercise}{RESET}")
            sys.exit(1)

    print(f"\n{BOLD}=== Moulinette — {args.module} ==={RESET}\n")
    passed = 0
    skipped = 0
    stopped = False
    for name, _fn, *opts in exercises:
        opts = opts[0] if opts else {}
        is_long = opts.get("long", False)
        timeout = opts.get("timeout", 60)
        if stopped:
            print(f"  {name}  {GREY}non évalué{RESET}")
            continue
        if is_long and not args.long:
            print(f"  {name}  {YELLOW}entraînement long : relance avec --long pour l'évaluer{RESET}")
            skipped += 1
            continue
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), args.module, name, "--_run"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            ok = proc.returncode == 0
            msg = (proc.stdout.strip().splitlines() or [""])[-1] if not ok else ""
            if not ok and not msg:
                msg = (proc.stderr.strip().splitlines() or ["erreur inconnue"])[-1]
        except subprocess.TimeoutExpired:
            ok, msg = False, f"timeout ({timeout}s) : boucle infinie ou code beaucoup trop lent ?"
        elapsed = time.time() - start
        if ok:
            passed += 1
            print(f"  {name}  {GREEN}OK{RESET}  {GREY}({elapsed:.1f}s){RESET}")
        else:
            print(f"  {name}  {RED}KO{RESET}  -> {msg}")
            stopped = True

    total = len(exercises) - skipped
    color = GREEN if passed == total else RED
    note = f"  {YELLOW}({skipped} non lancé){RESET}" if skipped else ""
    print(f"\n{BOLD}Résultat : {color}{passed}/{total}{RESET}{note}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
