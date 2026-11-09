"""Write docs/results.md from results/results.json (so the docs never disagree with the report)."""

from __future__ import annotations

import json

NAMES = {"qlearning_original": "Q-learning - Original", "sarsa_original": "SARSA - Original",
         "qlearning_modified": "Q-learning - Modified", "sarsa_modified": "SARSA - Modified"}


def main(path: str = "results/results.json", out: str = "docs/results.md") -> None:
    """Render the key tables as Markdown."""
    R = json.load(open(path, encoding="utf-8"))
    a = R["verification"]["random_audit"]
    L = ["# Results (generated from results/results.json)", "",
         f"Run: {R['meta']['timestamp']} on `{R['meta']['hostname']}` (Python {R['meta']['python']}, "
         f"Gymnasium {R['meta']['gymnasium']}), total {R['meta'].get('total_seconds', 0):.0f}s.", "",
         "## (a) Wrapper verification", "",
         f"* Attempted changes {a['attempted_changes']}, replaced {a['replaced']} -> rate {a['replacement_rate']:.4f} "
         f"(95% CI {a['ci95'][0]:.4f}-{a['ci95'][1]:.4f})",
         f"* Replacements when a == a_prev: {a['replaced_on_repeat']} of {a['repeat_steps']} steps",
         f"* Penalty errors: {a['penalty_on_selected_1']} (a=1 penalised), {a['penalty_missing_on_push']} (push not penalised)",
         f"* Bonus violations: {R['verification']['bonus_audit']['violations']}",
         f"* PDF example a_exec: {[r['a_exec'] for r in R['verification']['pdf_example']]}", "",
         "## (d) Greedy evaluation (100 episodes, epsilon = 0, seed 84)", "",
         "| Agent | mean reward | success | gentle | mean length | pushes | arrival v |", "|---|---|---|---|---|---|---|"]
    for k, v in R["evaluation"].items():
        L.append(f"| {NAMES[k]} | {v['mean_reward']:.2f} | {v['success_rate']:.2f} | {v['gentle_rate']:.2f} | "
                 f"{v['mean_length']:.1f} | {v['mean_pushes']:.1f} | {v['mean_arrival_velocity']:.4f} |")
    L += ["", "## Five seeds (84-88), greedy evaluation", "", "| Agent | reward mean +- std | success | gentle |", "|---|---|---|---|"]
    for k, v in R["multi_seed"]["summary"].items():
        L.append(f"| {NAMES[k]} | {v['mean_reward']['mean']:.1f} +- {v['mean_reward']['std']:.1f} | "
                 f"{v['success_rate']['mean']:.2f} | {v['gentle_rate']['mean']:.2f} |")
    gp = R["multi_seed"]["value_gap_phases"]
    L += ["", "## Value gap Q-learning - SARSA (mean over training, 5 seeds)", "",
          f"* original {gp['original']['all']['mean']:.2f} +- {gp['original']['all']['std']:.2f}; "
          f"modified {gp['modified']['all']['mean']:.2f} +- {gp['modified']['all']['std']:.2f}",
          f"* first 20% of training: original {gp['original']['first20pct']['mean']:.2f}, modified {gp['modified']['first20pct']['mean']:.2f}",
          "", "Figures: `figures/fig1_reward.png` ... `figures/fig9_length.png`."]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
