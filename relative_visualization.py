import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr, ttest_ind
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MultiModelCalibrationComparator:
    def __init__(self, gemma_path, medgemma_path):
        self.gemma_path = gemma_path
        self.medgemma_path = medgemma_path
        self.df_combined = None
        self.results = {}

    def load_and_combine_data(self):
        try:
            print('Loading data...')
            df_gemma = pd.read_csv(self.gemma_path)
            print(f'Gemma-3: {len(df_gemma)} samples')
            df_medgemma = pd.read_csv(self.medgemma_path)
            print(f'MedGemma: {len(df_medgemma)} samples')
            self.df_combined = pd.concat([df_gemma, df_medgemma], ignore_index=True)
            required_cols = ['Model', 'Prompt_Template', 'Confidence', 'Semantic_F1']
            missing_cols = [col for col in required_cols if col not in self.df_combined.columns]
            if missing_cols:
                print(f'Missing columns: {missing_cols}')
                return False
            self.df_combined['Confidence'] = pd.to_numeric(self.df_combined['Confidence'], errors='coerce')
            self.df_combined['Semantic_F1'] = pd.to_numeric(self.df_combined['Semantic_F1'], errors='coerce')
            self.df_combined = self.df_combined.dropna(subset=['Confidence', 'Semantic_F1'])
            print(f'Combined: {len(self.df_combined)} samples')
            print('Models:', self.df_combined['Model'].unique())
            print('Prompts:', self.df_combined['Prompt_Template'].unique())
            return True
        except Exception as e:
            print(e)
            return False

    def relative_calibration_error(self, confidence, performance, nbins=10):
        perf_min, perf_max = performance.min(), performance.max()
        if perf_max > perf_min:
            perf_normalized = (performance - perf_min) / (perf_max - perf_min)
        else:
            perf_normalized = np.zeros_like(performance)
        conf_normalized = confidence / 100.0
        bin_boundaries = np.linspace(0, 1, nbins + 1)
        rce = 0
        bin_details = []
        for i in range(nbins):
            in_bin = (conf_normalized >= bin_boundaries[i]) & (conf_normalized < bin_boundaries[i+1])
            if in_bin.sum() == 0:
                continue
            avg_conf_in_bin = conf_normalized[in_bin].mean()
            avg_perf_in_bin = perf_normalized[in_bin].mean()
            bin_proportion = in_bin.mean()
            bin_error = abs(avg_conf_in_bin - avg_perf_in_bin)
            rce += bin_error * bin_proportion
            bin_details.append({
                'bin_idx': i,
                'bin_proportion': bin_proportion,
                'avg_confidence': avg_conf_in_bin,
                'avg_performance': avg_perf_in_bin,
                'bin_error': bin_error,
                'sample_count': in_bin.sum()
            })
        return rce, bin_details

    def comprehensive_analysis(self):
        print('Two-Model Comparative Analysis...')
        results = {}
        for model in self.df_combined['Model'].unique():
            model_data = self.df_combined[self.df_combined['Model'] == model]
            confidence = model_data['Confidence']
            performance = model_data['Semantic_F1']
            conf_normalized = confidence / 100.0
            abs_gap = conf_normalized.mean() - performance.mean()
            abs_ece = abs_gap  # Simplified ECE for comparison
            pearson_r, pearson_p = pearsonr(confidence, performance)
            spearman_r, spearman_p = spearmanr(confidence, performance)
            kendall_tau, kendall_p = kendalltau(confidence, performance)
            # Percentile-based concordance
            conf_ranks = confidence.rank(pct=True)
            perf_ranks = performance.rank(pct=True)
            rank_correlation = pearsonr(conf_ranks, perf_ranks)[0]
            rce, bin_details = self.relative_calibration_error(confidence, performance)
            # Concordance rate
            npairs = min(len(confidence), 1000)
            concordant_pairs = 0
            total_comparisons = 0
            if len(confidence) > 1:
                sample_indices = np.random.choice(len(confidence), npairs, replace=False)
                for i, idx_i in enumerate(sample_indices):
                    for j, idx_j in enumerate(sample_indices[i+1:], i+1):
                        conf_diff = confidence.iloc[idx_i] - confidence.iloc[idx_j]
                        perf_diff = performance.iloc[idx_i] - performance.iloc[idx_j]
                        if abs(conf_diff) < 0.01:
                            continue
                        total_comparisons += 1
                        if conf_diff * perf_diff > 0:
                            concordant_pairs += 1
                concordance_rate = concordant_pairs / total_comparisons if total_comparisons > 0 else 0
            else:
                concordance_rate = 0
            # Percentile gap
            conf_percentiles = confidence.rank(pct=True) * 100
            perf_percentiles = performance.rank(pct=True) * 100
            percentile_gap = conf_percentiles - perf_percentiles
            results[model] = {
                'nsamples': len(model_data),
                'mean_confidence': confidence.mean(),
                'std_confidence': confidence.std(),
                'mean_f1': performance.mean(),
                'std_f1': performance.std(),
                'absolute_calibration_gap': abs_gap,
                'absolute_ece': abs_ece,
                'pearsonr': pearson_r,
                'pearsonp': pearson_p,
                'spearmanr': spearman_r,
                'spearmanp': spearman_p,
                'kendalltau': kendall_tau,
                'kendallp': kendall_p,
                'rank_correlation': rank_correlation,
                'relative_calibration_error': rce,
                'concordance_rate': concordance_rate,
                'mean_percentile_gap': percentile_gap.mean(),
                'std_percentile_gap': percentile_gap.std(),
                'percentile_gap_q25': percentile_gap.quantile(0.25),
                'percentile_gap_q75': percentile_gap.quantile(0.75),
                'bin_details': bin_details
            }
            print(f'{model}: Gap {abs_gap:.3f}, RCE {rce:.3f}')
            print(f'Pearson {pearson_r:.3f}, Spearman {spearman_r:.3f}')
        self.results['comparative'] = pd.DataFrame(results).T
        return self.results['comparative']

    def create_comparison_plots(self):
        print('Creating comparison plots...')
        models = list(self.df_combined['Model'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))
        fig = plt.figure(figsize=(16, 8))

        # 1. Confidence vs Performance Scatter
        ax1 = plt.subplot(2, 2, 1)
        ax1.text(-0.18, 1.10, '(A)', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            ax1.scatter(model_data['Confidence'], model_data['Semantic_F1'], alpha=0.6, label=model, s=20, color=model_colors[model])
        ax1.plot([0, 100], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Semantic F1')
        ax1.set_title('Absolute Confidence vs Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Calibration Gap Boxplot
        ax2 = plt.subplot(2, 2, 2)
        ax2.text(-0.18, 1.10, '(B)', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
        gap_data = []
        gap_labels = []
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            calibration_gap = model_data['Confidence'] / 100.0 - model_data['Semantic_F1']
            gap_data.append(calibration_gap.values)
            gap_labels.append(model)
        ax2.boxplot(gap_data, labels=gap_labels, patch_artist=True)
        for patch, color in zip(ax2.artists, model_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Absolute Calibration Gap')
        ax2.set_title('Calibration Gap Comparison')
        ax2.grid(True, alpha=0.3)

        # 3. Reliability Diagram
        ax3 = plt.subplot(2, 2, 3)
        ax3.text(-0.18, 1.10, '(C)', transform=ax3.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
        for model in models:
            if 'comparative' in self.results:
                model_results = self.results['comparative'].loc[model]
                bin_details = model_results['bin_details']
                bin_confs = [bd['avg_confidence'] for bd in bin_details if bd['sample_count'] > 0]
                bin_perfs = [bd['avg_performance'] for bd in bin_details if bd['sample_count'] > 0]
                ax3.plot(bin_confs, bin_perfs, 'o-', label=model, color=model_colors[model], linewidth=2, markersize=6)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax3.set_xlabel('Normalized Confidence')
        ax3.set_ylabel('Normalized Performance')
        ax3.set_title('Reliability Diagram')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Correlation Coefficient Comparison
        ax4 = plt.subplot(2, 2, 4)
        ax4.text(-0.18, 1.10, '(D)', transform=ax4.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
        if 'comparative' in self.results:
            comp_df = self.results['comparative']
            x = np.arange(len(models))
            width = 0.25
            pearson_rs = comp_df['pearsonr']
            spearman_rs = comp_df['spearmanr']
            kendall_taus = comp_df['kendalltau']
            ax4.bar(x - width, pearson_rs, width, label='Pearson r', alpha=0.8)
            ax4.bar(x, spearman_rs, width, label='Spearman', alpha=0.8)
            ax4.bar(x + width, kendall_taus, width, label="Kendall's τ", alpha=0.8)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Correlation Coefficient')
            ax4.set_title('Correlation Coefficient Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(models, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('multimodel_calibration_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print('Comparison plots saved as multimodel_calibration_comparison.png')

        # Supplementary: Prompt별 Calibration Gap Boxplot
        plt.figure(figsize=(10,6))
        sns.boxplot(
            data=self.df_combined,
            x='Prompt_Template',
            y=self.df_combined['Confidence']/100.0 - self.df_combined['Semantic_F1'],
            hue='Model',
            palette='Set2'
        )
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('Calibration Gap')
        plt.title('Calibration Gap by Prompt Template and Model')
        plt.tight_layout()
        plt.savefig('promptwise_calibration_gap.png', dpi=300)
        plt.show()

        prompt_metrics = []
        for model in self.df_combined['Model'].unique():
            for prompt in self.df_combined['Prompt_Template'].unique():
                subset = self.df_combined[(self.df_combined['Model'] == model) & (self.df_combined['Prompt_Template'] == prompt)]
                if subset.empty:
                    continue
                confidence = subset['Confidence']
                performance = subset['Semantic_F1']
                # ECE 계산
                conf_normalized = confidence / 100.0
                nbins = 10
                bin_boundaries = np.linspace(0, 1, nbins + 1)
                ece = 0
                for i in range(nbins):
                    in_bin = (conf_normalized >= bin_boundaries[i]) & (conf_normalized < bin_boundaries[i+1])
                    if in_bin.sum() == 0:
                        continue
                    avg_conf = conf_normalized[in_bin].mean()
                    avg_perf = performance[in_bin].mean()
                    bin_error = abs(avg_conf - avg_perf)
                    ece += bin_error * in_bin.mean()
                # RCE (standardized for each prompt)
                perf_min, perf_max = performance.min(), performance.max()
                perf_norm = (performance - perf_min) / (perf_max - perf_min) if perf_max > perf_min else np.zeros_like(performance)
                rce = 0
                for i in range(nbins):
                    in_bin = (conf_normalized >= bin_boundaries[i]) & (conf_normalized < bin_boundaries[i+1])
                    if in_bin.sum() == 0:
                        continue
                    avg_conf = conf_normalized[in_bin].mean()
                    avg_perf = perf_norm[in_bin].mean()
                    bin_error = abs(avg_conf - avg_perf)
                    rce += bin_error * in_bin.mean()
                # correlation (Spearman's rho)
                try:
                    corr, _ = spearmanr(confidence, performance)
                except:
                    corr = np.nan
                prompt_metrics.append({
                    'Model': model,
                    'Prompt': prompt,
                    'ECE': ece,
                    'RCE': rce,
                    'SpearmanRho': corr
                })

        metrics_df = pd.DataFrame(prompt_metrics)

        plt.figure(figsize=(16, 5))
        plt.figure(figsize=(16, 5))

        # 1. ECE
        ax1 = plt.subplot(1, 3, 1)
        sns.barplot(data=metrics_df, x='Prompt', y='ECE', hue='Model', ax=ax1)
        ax1.set_title('Prompt-wise ECE')
        ax1.axhline(0.15, linestyle='--', color='red', label='Clinical Threshold')
        ax1.legend()
        ax1.text(-0.18, 1.10, '(A)', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        # 2. RCE
        ax2 = plt.subplot(1, 3, 2)
        sns.barplot(data=metrics_df, x='Prompt', y='RCE', hue='Model', ax=ax2)
        ax2.set_title('Prompt-wise RCE')
        ax2.legend()
        ax2.text(-0.18, 1.10, '(B)', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        # 3. Spearman Correlation
        ax3 = plt.subplot(1, 3, 3)
        sns.barplot(data=metrics_df, x='Prompt', y='SpearmanRho', hue='Model', ax=ax3)
        ax3.set_title('Prompt-wise Confidence-Performance Correlation')
        ax3.axhline(0, linestyle='--', color='gray')
        ax3.legend()
        ax3.text(-0.18, 1.10, '(C)', transform=ax3.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        plt.suptitle('Prompt Template Reliability Metrics')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('promptwise_reliability_metrics.png', dpi=300)
        plt.show()



if __name__ == "__main__":
    GEMMA_PATH = '/Users/sohyeon/Downloads/VitalLab/LLM_Cognitive_Abilities/data/analysis_details_integrated_gemma3.csv'
    MEDGEMMA_PATH = '/Users/sohyeon/Downloads/VitalLab/LLM_Cognitive_Abilities/data/analysis_details_integrated_medgemma.csv'
    comparator = MultiModelCalibrationComparator(GEMMA_PATH, MEDGEMMA_PATH)
    if comparator.load_and_combine_data():
        comparator.comprehensive_analysis()
        comparator.create_comparison_plots()
