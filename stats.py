"""Statistical Analysis Module for CVRP Methods Comparison.

Provides:
- Descriptive statistics (Mean, Std, Median, IQR, Min, Max, Gap %)
- Normality Tests (Shapiro-Wilk)
- Omnibus Paired Hypothesis Tests (Friedman Test)
- Pairwise Post-Hoc Tests (Paired Wilcoxon Signed-Rank Test & Dunn's Test)
- Multiple Comparisons Correction (Holm-Bonferroni, Bonferroni, FDR Benjamini-Hochberg)
- Effect Size (Rank-Biserial Correlation, Cliff's Delta)
- Markdown and LaTeX table generation for academic papers / dissertation
"""

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without requiring external tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        formatted_row = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_row.append(f"{val:.4f}" if (abs(val) >= 0.0001 or val == 0) else f"{val:.4e}")
            else:
                formatted_row.append(str(val))
        rows.append(formatted_row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_str = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    body_str = "\n".join(
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows
    )

    return f"{header_str}\n{separator_str}\n{body_str}"


def df_to_latex_table(
    df: pd.DataFrame, caption: Optional[str] = None, label: Optional[str] = None
) -> str:
    """Format DataFrame as a LaTeX tabular / table without jinja2 dependency."""
    if df.empty:
        return ""
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")

    headers_escaped = [c.replace("_", r"\_").replace("%", r"\%") for c in cols]
    lines.append(" & ".join(headers_escaped) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        row_vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted = f"{val:.4f}" if (abs(val) >= 0.0001 or val == 0) else f"{val:.4e}"
            else:
                formatted = str(val).replace("_", r"\_").replace("%", r"\%")
            row_vals.append(formatted)
        lines.append(" & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni sequential step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj)
        adjusted_p[orig_idx] = min(1.0, running_max)

    return adjusted_p


def fdr_benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) correction."""
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m

    values = []
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        q = (p_val * m) / (rank + 1)
        values.append((orig_idx, q))

    running_min = 1.0
    for orig_idx, q in reversed(values):
        running_min = min(running_min, q)
        adjusted_p[orig_idx] = min(1.0, max(0.0, running_min))

    return adjusted_p


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Rank-Biserial Correlation effect size for paired samples (x vs y)."""
    diffs = x - y
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(diffs))
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    total_w = w_pos + w_neg
    if total_w == 0:
        return 0.0

    return float((w_pos - w_neg) / total_w)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's Delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    greater = np.sum(x[:, None] > y[None, :])
    smaller = np.sum(x[:, None] < y[None, :])
    return float((greater - smaller) / (n_x * n_y))


class StatisticalAnalyzer:
    """Statistical analyzer comparing CVRP heuristic and optimization algorithms."""

    def __init__(
        self,
        data: pd.DataFrame,
        metric: str = "distance",
        instance_col: str = "instance",
        method_col: str = "method",
    ):
        """Initialize with a results DataFrame."""
        self.raw_data = data.copy()
        self.metric = metric
        self.instance_col = instance_col
        self.method_col = method_col

        # Deduplicate: if multiple runs exist for the same (instance, method), keep latest
        dedup_data = self.raw_data.drop_duplicates(subset=[self.instance_col, self.method_col], keep="last")

        # Pivot to create paired instances x methods table
        self.pivot_df = dedup_data.pivot(
            index=self.instance_col, columns=self.method_col, values=self.metric
        ).dropna()

        self.methods = list(self.pivot_df.columns)
        self.n_instances = len(self.pivot_df)

    @classmethod
    def from_csv_files(
        cls, csv_paths: List[Union[str, Path]], metric: str = "distance", by_run: bool = False
    ) -> "StatisticalAnalyzer":
        """Load and merge multiple results.csv files."""
        dfs = []
        for p in csv_paths:
            path = Path(p)
            if path.is_file():
                df = pd.read_csv(path)
                parent = path.parent

                # Check if params.json exists to get precise method/config
                method_name = None
                params_file = parent / "params.json"
                if params_file.is_file():
                    try:
                        with open(params_file) as f:
                            params_data = json.load(f)
                            method_name = params_data.get("method")
                    except Exception:
                        pass

                if by_run:
                    df["method"] = parent.name
                else:
                    if method_name:
                        df["method"] = method_name.upper()
                    elif "method" not in df.columns:
                        folder_parts = parent.name.split("_")
                        df["method"] = folder_parts[2].upper() if len(folder_parts) >= 3 else parent.name

                dfs.append(df)

        if not dfs:
            raise FileNotFoundError("No valid CSV files provided.")

        merged_df = pd.concat(dfs, ignore_index=True)
        return cls(merged_df, metric=metric)

    @classmethod
    def from_solutions_dir(
        cls, solutions_dir: Union[str, Path], metric: str = "distance", by_run: bool = False
    ) -> "StatisticalAnalyzer":
        """Scan solutions directory and load all results.csv files sorted chronologically."""
        sol_dir = Path(solutions_dir)
        csv_files = sorted(list(sol_dir.glob("**/results.csv")), key=lambda p: p.stat().st_mtime)
        if not csv_files:
            raise FileNotFoundError(f"No results.csv found under {solutions_dir}")
        return cls.from_csv_files(csv_files, metric=metric, by_run=by_run)

    def descriptive_statistics(self) -> pd.DataFrame:
        """Calculate descriptive statistics for each method."""
        stats_list = []
        min_per_instance = self.pivot_df.min(axis=1)

        for m in self.methods:
            vals = self.pivot_df[m]
            gaps = ((vals - min_per_instance) / min_per_instance) * 100.0

            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)

            stats_list.append({
                "Method": m,
                "Count": len(vals),
                "Mean": vals.mean(),
                "Std": vals.std(),
                "Median": vals.median(),
                "IQR": q75 - q25,
                "Min": vals.min(),
                "Max": vals.max(),
                "Avg_Gap_%": gaps.mean(),
            })

        return pd.DataFrame(stats_list)

    def normality_tests(self) -> pd.DataFrame:
        """Execute Shapiro-Wilk normality test for each method and pairwise differences."""
        rows = []
        for m in self.methods:
            vals = self.pivot_df[m].to_numpy()
            if len(vals) >= 3:
                stat, p_val = stats.shapiro(vals)
                rows.append({
                    "Target": f"{m} (Distribution)",
                    "W_Stat": round(stat, 5),
                    "p_value": p_val,
                    "Normal (p >= 0.05)": p_val >= 0.05,
                })

        for m1, m2 in combinations(self.methods, 2):
            diff = (self.pivot_df[m1] - self.pivot_df[m2]).to_numpy()
            if len(diff) >= 3:
                stat, p_val = stats.shapiro(diff)
                rows.append({
                    "Target": f"Diff ({m1} - {m2})",
                    "W_Stat": round(stat, 5),
                    "p_value": p_val,
                    "Normal (p >= 0.05)": p_val >= 0.05,
                })

        return pd.DataFrame(rows)

    def omnibus_test(self) -> Dict[str, Union[str, float, int]]:
        """Run omnibus test (Friedman test for >=3 methods, Wilcoxon for 2 methods)."""
        if len(self.methods) < 2:
            return {"error": "At least 2 methods required for comparison."}

        if len(self.methods) == 2:
            m1, m2 = self.methods
            stat, p_val = stats.wilcoxon(self.pivot_df[m1], self.pivot_df[m2])
            return {
                "Test": "Wilcoxon Signed-Rank Test",
                "Statistic": round(float(stat), 4),
                "p_value": float(p_val),
                "Significant (alpha=0.05)": p_val < 0.05,
            }

        data_arrays = [self.pivot_df[m].to_numpy() for m in self.methods]
        friedman_stat, friedman_p = stats.friedmanchisquare(*data_arrays)

        return {
            "Test": "Friedman Test",
            "Chi2_Statistic": round(float(friedman_stat), 4),
            "DoF": len(self.methods) - 1,
            "p_value": float(friedman_p),
            "Significant (alpha=0.05)": friedman_p < 0.05,
        }

    def pairwise_posthoc(self) -> pd.DataFrame:
        """Run pairwise Wilcoxon signed-rank and Dunn-style comparisons with Holm-Bonferroni correction."""
        pairs = list(combinations(self.methods, 2))
        if not pairs:
            return pd.DataFrame()

        ranks_matrix = self.pivot_df.rank(axis=1, ascending=True)
        mean_ranks = ranks_matrix.mean(axis=0)

        raw_results = []
        wilcoxon_p_values = []

        for m1, m2 in pairs:
            v1 = self.pivot_df[m1].to_numpy()
            v2 = self.pivot_df[m2].to_numpy()
            diff = v1 - v2

            mean_diff = float(np.mean(diff))
            median_diff = float(np.median(diff))

            try:
                w_stat, w_pval = stats.wilcoxon(v1, v2)
            except Exception:
                w_stat, w_pval = 0.0, 1.0

            wilcoxon_p_values.append(w_pval)

            rb_corr = rank_biserial_correlation(v1, v2)
            c_delta = cliffs_delta(v1, v2)

            rank_diff = mean_ranks[m1] - mean_ranks[m2]
            se = np.sqrt((len(self.methods) * (len(self.methods) + 1)) / (6 * self.n_instances))
            dunn_z = rank_diff / se if se > 0 else 0.0
            dunn_p = 2 * (1 - stats.norm.cdf(abs(dunn_z)))

            raw_results.append({
                "Comparison": f"{m1} vs {m2}",
                "Mean_Diff": mean_diff,
                "Median_Diff": median_diff,
                "Rank_Diff": rank_diff,
                "Wilcoxon_W": w_stat,
                "p_raw": w_pval,
                "Dunn_Z": dunn_z,
                "Dunn_p": dunn_p,
                "Rank_Biserial": rb_corr,
                "Cliffs_Delta": c_delta,
            })

        holm_p = holm_bonferroni_correction(wilcoxon_p_values)
        fdr_p = fdr_benjamini_hochberg(wilcoxon_p_values)
        bonferroni_p = [min(1.0, p * len(pairs)) for p in wilcoxon_p_values]

        for i, row in enumerate(raw_results):
            row["p_Holm"] = holm_p[i]
            row["p_Bonf"] = bonferroni_p[i]
            row["p_FDR"] = fdr_p[i]
            row["Sig (Holm < 0.05)"] = "Yes *" if holm_p[i] < 0.05 else "No"
            row["Winner"] = (
                row["Comparison"].split(" vs ")[0]
                if row["Mean_Diff"] < 0 and holm_p[i] < 0.05
                else (
                    row["Comparison"].split(" vs ")[1]
                    if row["Mean_Diff"] > 0 and holm_p[i] < 0.05
                    else "Tie / Inconclusive"
                )
            )

        return pd.DataFrame(raw_results)

    def generate_latex_tables(self) -> str:
        """Generate ready-to-use LaTeX tables for academic papers / dissertation."""
        desc_df = self.descriptive_statistics()
        posthoc_df = self.pairwise_posthoc()

        latex_str = []
        latex_str.append("% --- Tabela de Estatísticas Descritivas ---")
        latex_str.append(df_to_latex_table(desc_df, caption="Estatísticas Descritivas dos Métodos de Roteamento."))
        latex_str.append("\n% --- Tabela de Testes Post-Hoc com Correção de Holm ---")
        cols_to_latex = [
            "Comparison", "Mean_Diff", "Median_Diff", "Wilcoxon_W", "p_raw", "p_Holm", "Rank_Biserial", "Winner"
        ]
        if not posthoc_df.empty:
            latex_str.append(
                df_to_latex_table(posthoc_df[cols_to_latex], caption="Testes Pareados Post-Hoc de Wilcoxon com Correção de Holm.")
            )

        return "\n".join(latex_str)

    def summary_report(self) -> str:
        """Generate a complete formatted Markdown report."""
        if self.pivot_df.empty:
            return "⚠️ Nenhuma instância em comum encontrada entre os métodos selecionados para comparação pareada."

        desc_df = self.descriptive_statistics()
        norm_df = self.normality_tests()
        omnibus = self.omnibus_test()
        posthoc_df = self.pairwise_posthoc()

        lines = []
        lines.append(f"# 📊 Relatório de Análise Estatística ({self.metric.upper()})\n")
        lines.append(f"- **Total de instâncias pareadas avaliadas:** {self.n_instances}")
        lines.append(f"- **Métodos avaliados:** {', '.join(self.methods)}\n")

        lines.append("## 1. Estatísticas Descritivas")
        lines.append(df_to_markdown_table(desc_df))
        lines.append("")

        lines.append("## 2. Teste de Normalidade (Shapiro-Wilk)")
        lines.append(df_to_markdown_table(norm_df))
        lines.append("")

        lines.append("## 3. Teste Hipotético Global (Omnibus)")
        for k, v in omnibus.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

        lines.append("## 4. Testes Pareados Post-Hoc com Correção de Holm")
        if not posthoc_df.empty:
            display_cols = [
                "Comparison", "Mean_Diff", "Median_Diff", "p_raw", "p_Holm", "p_Bonf", "Rank_Biserial", "Sig (Holm < 0.05)", "Winner"
            ]
            lines.append(df_to_markdown_table(posthoc_df[display_cols]))
        else:
            lines.append("Apenas 1 método presente nos dados.")
        lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    # Ensure project root is in sys.path
    project_root = str(Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "models.py").exists() else Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Statistical Analysis of CVRP Method Results")
    parser.add_argument("--solutions_dir", type=str, default="solutions", help="Directory containing solution run folders with results.csv")
    parser.add_argument("--csv", nargs="+", help="Specific results.csv files to compare")
    parser.add_argument("--metric", type=str, default="distance", choices=["distance", "time_online", "routes"], help="Metric to analyze")
    parser.add_argument("--by_run", action="store_true", help="Compare each execution folder individually instead of aggregating by method")
    parser.add_argument("--export_latex", action="store_true", help="Print LaTeX table code")

    args = parser.parse_args()

    if args.csv:
        analyzer = StatisticalAnalyzer.from_csv_files(args.csv, metric=args.metric, by_run=args.by_run)
    else:
        analyzer = StatisticalAnalyzer.from_solutions_dir(args.solutions_dir, metric=args.metric, by_run=args.by_run)

    report = analyzer.summary_report()
    print(report)

    if args.export_latex:
        print("\n" + "=" * 60)
        print("LATEX CODE:")
        print("=" * 60)
        print(analyzer.generate_latex_tables())
