import json
import os

input_json = "STATISTICAL_VALIDATION_FINAL.json"
output_tex = "paper/ablation_table.tex"

if not os.path.exists(input_json):
    print(f"Error: {input_json} not found.")
    exit(1)

with open(input_json, "r") as f:
    data = json.load(f)

# LaTeX Table Header
latex_str = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Component Ablation and Statistical Validation (10,000 Permutations)}
\label{tab:ablation}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}llccrrr@{}}
\toprule
\textbf{ID} & \textbf{Ablated Component} & \textbf{Evaluation Metric} & \textbf{Baseline} & \textbf{Ablated} & \textbf{Effect Size ($d$)} & \textbf{$p$-value} \\ \midrule
"""

def clean_name(name):
    # e.g., A1_Remove_Signatures -> Remove Signatures
    parts = name.split('_', 1)
    if len(parts) > 1:
        return parts[1].replace("_", " ")
    return name

for key, metrics in data.items():
    ablation_id = key.split('_')[0]
    comp_name = clean_name(key)
    
    # Determine the metric used
    metric_name = "Classification F1"
    b_val = metrics.get("baseline_f1", 0.0)
    a_val = metrics.get("ablated_f1", 0.0)
    
    if "baseline_lateral_movement_f1" in metrics:
        metric_name = "Lateral Movement F1"
        b_val = metrics.get("baseline_lateral_movement_f1", 0.0)
        a_val = metrics.get("ablated_lateral_movement_f1", 0.0)
    elif "baseline_brier_score" in metrics:
        metric_name = "Brier Score (Calibration)"
        b_val = metrics.get("baseline_brier_score", 0.0)
        a_val = metrics.get("ablated_brier_score", 0.0)
        
    effect_size = metrics.get("effect_size", 0.0)
    p_val = metrics.get("adjusted_p", metrics.get("raw_p", 0.0))
    
    # Format values
    b_str = f"{b_val:.3f}"
    a_str = f"{a_val:.3f}"
    d_str = f"{effect_size:.3f}"
    
    if p_val < 0.001:
        p_str = "$< 0.001$*"
    else:
        p_str = f"{p_val:.3f}"
        
    # Highlight paradoxical improvements
    if metric_name == "Classification F1" and effect_size < -0.1 and (a_val > b_val):
        d_str = f"\\textbf{{{d_str}}}" # Highlight inverse ablations
        
    row = f"{ablation_id} & {comp_name} & {metric_name} & {b_str} & {a_str} & {d_str} & {p_str} \\\\\n"
    latex_str += row

# LaTeX Table Footer
latex_str += r"""\bottomrule
\end{tabular}
}
\vspace{1ex}
{\raggedright \footnotesize * Indicates statistical significance after Holm-Bonferroni correction ($\alpha=0.05$). Bold effect sizes denote regulatory modules where ablation artificially inflates static classification F1 at the cost of operational safety (see Section VI). \par}
\end{table*}
"""

os.makedirs("paper", exist_ok=True)
with open(output_tex, "w") as f:
    f.write(latex_str)

print(f"LaTeX table successfully generated at {output_tex}")
