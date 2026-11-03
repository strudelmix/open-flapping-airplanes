"""Offline inspection of the planned B200 run matrix; requires only Python 3."""
import argparse


def plan(steps=600, repeats=2, full_steps=4000, explore_one_sm=False):
    if steps < 20 or repeats < 2 or full_steps < steps:
        raise ValueError("steps >= 20, repeats >= 2 and full_steps >= steps required")
    yield ("correctness", "HR128", 100, "no optimizer, no Doom, greedy")
    for hr in (64, 256):
        yield ("correctness", f"HR{hr}", 100, "compare shared greedy-token prefix and loss")
    for rep in range(repeats):
        for hr in (128, 64, 256):
            yield ("quick", f"HR{hr}/Doom8/r{rep + 1}", steps, "fixed train116/infer24/doom8")
    for doom in (4, 2) + ((1,) if explore_one_sm else ()):
        for rep in range(repeats):
            yield ("sweep", f"best-HR/Doom{doom}/r{rep + 1}", steps, f"train {124-doom}, infer24, doom{doom}")
    yield ("confirm", "HR128/Doom8", full_steps, "B200 baseline comparison")
    yield ("confirm", "candidate", full_steps, "finalist, if different")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--full-steps", type=int, default=4000)
    parser.add_argument("--explore-one-sm", action="store_true")
    args = parser.parse_args()
    for group, name, steps, reason in plan(args.steps, args.repeats, args.full_steps, args.explore_one_sm):
        print(f"{group:12} {name:26} {steps:5d} steps  {reason}")
    print("No GPU was rented or tested by this command.")
