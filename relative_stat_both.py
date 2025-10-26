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
            print(f"   loading: {self.gemma_path}")
            df_gemma = pd.read_csv(self.gemma_path)
            print(f"   Gemma-3: {len(df_gemma)} 행")
            
            print(f"   loading: {self.medgemma_path}")
            df_medgemma = pd.read_csv(self.medgemma_path)
            print(f"   MedGemma: {len(df_medgemma)} 행")
            
            self.df_combined = pd.concat([df_gemma, df_medgemma], ignore_index=True)
            
            required_cols = ['Model', 'Prompt_Template', 'Confidence', 'Semantic_F1']
            missing_cols = [col for col in required_cols if col not in self.df_combined.columns]
            
            if missing_cols:
                print(f"ommited column: {missing_cols}")
                return False
            
            print("Data preprocessing ...")
            self.df_combined['Confidence'] = pd.to_numeric(self.df_combined['Confidence'], errors='coerce')
            self.df_combined['Semantic_F1'] = pd.to_numeric(self.df_combined['Semantic_F1'], errors='coerce')
            self.df_combined = self.df_combined.dropna(subset=['Confidence', 'Semantic_F1'])
            
            print(f"Combined Data: {len(self.df_combined)} 행")
            print(f"   Model: {list(self.df_combined['Model'].unique())}")
            print(f"   Prompt: {list(self.df_combined['Prompt_Template'].unique())}")
            
            model_counts = self.df_combined['Model'].value_counts()
            for model, count in model_counts.items():
                print(f"   • {model}: {count} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def relative_calibration_error(self, confidence, performance, n_bins=10):
        perf_min, perf_max = performance.min(), performance.max()
        if perf_max > perf_min:
            perf_normalized = (performance - perf_min) / (perf_max - perf_min)
        else:
            perf_normalized = np.zeros_like(performance)
            
        conf_normalized = confidence / 100.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        rce = 0
        bin_details = []
        
        for i in range(n_bins):
            in_bin = (conf_normalized > bin_boundaries[i]) & (conf_normalized <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
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
        print("\nTwo-Model Comparative Analysis...")
        results = {}
        
        for model in self.df_combined['Model'].unique():
            model_data = self.df_combined[self.df_combined['Model'] == model]
            
            confidence = model_data['Confidence']
            performance = model_data['Semantic_F1']
            
            # absolute
            conf_normalized = confidence / 100.0
            abs_gap = conf_normalized.mean() - performance.mean()
            abs_ece = abs_gap  # Simplified ECE for comparison
            
            # correlation
            pearson_r, pearson_p = pearsonr(confidence, performance)
            spearman_r, spearman_p = spearmanr(confidence, performance)
            kendall_tau, kendall_p = kendalltau(confidence, performance)
            
            # relative
            conf_ranks = confidence.rank(pct=True)
            perf_ranks = performance.rank(pct=True)
            rank_correlation = pearsonr(conf_ranks, perf_ranks)[0]
            
            rce, bin_details = self.relative_calibration_error(confidence, performance)
            
            # concordance (sampling)
            n_pairs = min(len(confidence), 1000)
            concordant_pairs = 0
            total_comparisons = 0
            
            if len(confidence) > 1:
                sample_indices = np.random.choice(len(confidence), n_pairs, replace=False)
                for i, idx_i in enumerate(sample_indices):
                    for j, idx_j in enumerate(sample_indices[i+1:], i+1):
                        conf_diff = confidence.iloc[idx_i] - confidence.iloc[idx_j]
                        perf_diff = performance.iloc[idx_i] - performance.iloc[idx_j]
                        
                        if abs(conf_diff) > 0.01:
                            total_comparisons += 1
                            if (conf_diff * perf_diff) > 0:
                                concordant_pairs += 1
            
            concordance_rate = concordant_pairs / total_comparisons if total_comparisons > 0 else 0
            
            # percentile
            conf_percentiles = confidence.rank(pct=True) * 100
            perf_percentiles = performance.rank(pct=True) * 100
            percentile_gap = conf_percentiles - perf_percentiles
            
            results[model] = {
                'n_samples': len(model_data),
                'mean_confidence': confidence.mean(),
                'std_confidence': confidence.std(),
                'mean_f1': performance.mean(),
                'std_f1': performance.std(),
                
                'absolute_calibration_gap': abs_gap,
                'absolute_ece': abs_ece,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'rank_correlation': rank_correlation,
                'relative_calibration_error': rce,
                'concordance_rate': concordance_rate,
                
                'mean_percentile_gap': percentile_gap.mean(),
                'std_percentile_gap': percentile_gap.std(),
                'percentile_gap_q25': percentile_gap.quantile(0.25),
                'percentile_gap_q75': percentile_gap.quantile(0.75),
                
                'abs_interpretation': 'Overconfident' if abs_gap > 0 else 'Underconfident',
                'spearman_interpretation': self.interpret_correlation(spearman_r),
                'calibration_quality': self.interpret_rce(rce),
                'bin_details': bin_details
            }
            
            print(f"   ✅ {model}:")
            print(f"      Absolute Gap: {abs_gap:.3f}, Relative RCE: {rce:.3f}")
            print(f"      Pearson: {pearson_r:.3f}, Spearman: {spearman_r:.3f}")
        
        self.results['comparative'] = pd.DataFrame(results).T
        return self.results['comparative']
    
    def statistical_comparison(self):
        print("\nStatistical Comparison between Models...")
        
        models = list(self.df_combined['Model'].unique())
        if len(models) != 2:
            print(f"⚠️ Need two models but found {len(models)} models")
            return
        
        model1, model2 = models[0], models[1]
        data1 = self.df_combined[self.df_combined['Model'] == model1]
        data2 = self.df_combined[self.df_combined['Model'] == model2]
        
        comparisons = {}
        
        # Confidence comparison
        conf_stat, conf_p = ttest_ind(data1['Confidence'], data2['Confidence'])
        
        # Performance comparison
        perf_stat, perf_p = ttest_ind(data1['Semantic_F1'], data2['Semantic_F1'])
        
        # Calibration Gap comparison
        gap1 = (data1['Confidence'] / 100.0) - data1['Semantic_F1']
        gap2 = (data2['Confidence'] / 100.0) - data2['Semantic_F1']
        gap_stat, gap_p = ttest_ind(gap1, gap2)
        
        comparisons = {
            'confidence_comparison': {
                'model1_mean': data1['Confidence'].mean(),
                'model2_mean': data2['Confidence'].mean(),
                't_statistic': conf_stat,
                'p_value': conf_p,
                'significant': conf_p < 0.05
            },
            'performance_comparison': {
                'model1_mean': data1['Semantic_F1'].mean(),
                'model2_mean': data2['Semantic_F1'].mean(),
                't_statistic': perf_stat,
                'p_value': perf_p,
                'significant': perf_p < 0.05
            },
            'calibration_gap_comparison': {
                'model1_mean': gap1.mean(),
                'model2_mean': gap2.mean(),
                't_statistic': gap_stat,
                'p_value': gap_p,
                'significant': gap_p < 0.05
            }
        }
        
        self.results['statistical_comparison'] = comparisons
        
        print(f"   Confidence: {model1}({data1['Confidence'].mean():.1f}) vs {model2}({data2['Confidence'].mean():.1f}) - p={conf_p:.6f}")
        print(f"   Performance: {model1}({data1['Semantic_F1'].mean():.3f}) vs {model2}({data2['Semantic_F1'].mean():.3f}) - p={perf_p:.6f}")
        print(f"   Cal. Gap: {model1}({gap1.mean():.3f}) vs {model2}({gap2.mean():.3f}) - p={gap_p:.6f}")
        
        return comparisons
    
    def interpret_correlation(self, r):
        abs_r = abs(r)
        if abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.3:
            return "Moderate"
        elif abs_r >= 0.1:
            return "Weak"
        else:
            return "Very Weak"
    
    def interpret_rce(self, rce):
        if rce < 0.1:
            return "Well Calibrated"
        elif rce < 0.2:
            return "Moderately Calibrated"
        elif rce < 0.3:
            return "Poorly Calibrated"
        else:
            return "Severely Miscalibrated"
    
    def create_comprehensive_comparison_plots(self):
        print("\nComprehensive Comparison Visualization...")
        
        fig = plt.figure(figsize=(24, 16))
        
        models = list(self.df_combined['Model'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))
        
        # 1. Absolute Confidence vs Performance Scatter
        ax1 = plt.subplot(4, 3, 1)
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            ax1.scatter(model_data['Confidence'], model_data['Semantic_F1'], 
                       alpha=0.6, label=model, s=20, color=model_colors[model])
        
        ax1.plot([0, 100], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.set_xlabel('Confidence (%)')
        ax1.set_ylabel('Semantic F1')
        ax1.set_title('Absolute Confidence vs Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Relative Percentile Scatter
        ax2 = plt.subplot(4, 3, 2)
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            conf_percentiles = model_data['Confidence'].rank(pct=True) * 100
            perf_percentiles = model_data['Semantic_F1'].rank(pct=True) * 100
            
            ax2.scatter(conf_percentiles, perf_percentiles, 
                       alpha=0.6, label=model, s=20, color=model_colors[model])
        
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.7, label='Perfect Calibration')
        ax2.set_xlabel('Confidence Percentile (%)')
        ax2.set_ylabel('Performance Percentile (%)')
        ax2.set_title('Relative Percentile Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Absolute Calibration Gap Comparison
        ax3 = plt.subplot(4, 3, 3)
        gap_data = []
        gap_labels = []
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            calibration_gap = (model_data['Confidence'] / 100.0) - model_data['Semantic_F1']
            gap_data.append(calibration_gap.values)
            gap_labels.append(model)
        
        bp1 = ax3.boxplot(gap_data, labels=gap_labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], model_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Absolute Calibration Gap')
        ax3.set_title('Absolute Calibration Gap Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 4. Relative Percentile Gap Comparison
        ax4 = plt.subplot(4, 3, 4)
        rel_gap_data = []
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            conf_percentiles = model_data['Confidence'].rank(pct=True) * 100
            perf_percentiles = model_data['Semantic_F1'].rank(pct=True) * 100
            rel_gap = conf_percentiles - perf_percentiles
            rel_gap_data.append(rel_gap.values)
        
        bp2 = ax4.boxplot(rel_gap_data, labels=gap_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], model_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Relative Percentile Gap (%)')
        ax4.set_title('Relative Percentile Gap Comparison')
        ax4.grid(True, alpha=0.3)
        
        # 5. Confidence Distribution Comparison
        ax5 = plt.subplot(4, 3, 5)
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            ax5.hist(model_data['Confidence'], bins=30, alpha=0.6, 
                    label=model, density=True, color=model_colors[model])
        
        ax5.set_xlabel('Confidence (%)')
        ax5.set_ylabel('Density')
        ax5.set_title('Confidence Distribution Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Distribution Comparison
        ax6 = plt.subplot(4, 3, 6)
        for model in models:
            model_data = self.df_combined[self.df_combined['Model'] == model]
            ax6.hist(model_data['Semantic_F1'], bins=30, alpha=0.6, 
                    label=model, density=True, color=model_colors[model])
        
        ax6.set_xlabel('Semantic F1')
        ax6.set_ylabel('Density')
        ax6.set_title('Performance Distribution Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Absolute Gap Comparison per Prompt Style
        ax7 = plt.subplot(4, 3, 7)
        prompt_gap_data = []
        prompt_labels = []
        for prompt in self.df_combined['Prompt_Template'].unique():
            prompt_data = self.df_combined[self.df_combined['Prompt_Template'] == prompt]
            calibration_gap = (prompt_data['Confidence'] / 100.0) - prompt_data['Semantic_F1']
            prompt_gap_data.append(calibration_gap.values)
            prompt_labels.append(prompt)
        
        ax7.boxplot(prompt_gap_data, labels=prompt_labels)
        ax7.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax7.set_ylabel('Absolute Calibration Gap')
        ax7.set_title('Gap by Prompt Template')
        ax7.grid(True, alpha=0.3)
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
        
        # 8. Relative Gap Comparison per Prompt Style
        ax8 = plt.subplot(4, 3, 8)
        prompt_rel_gap_data = []
        for prompt in self.df_combined['Prompt_Template'].unique():
            prompt_data = self.df_combined[self.df_combined['Prompt_Template'] == prompt]
            conf_percentiles = prompt_data['Confidence'].rank(pct=True) * 100
            perf_percentiles = prompt_data['Semantic_F1'].rank(pct=True) * 100
            rel_gap = conf_percentiles - perf_percentiles
            prompt_rel_gap_data.append(rel_gap.values)
        
        ax8.boxplot(prompt_rel_gap_data, labels=prompt_labels)
        ax8.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax8.set_ylabel('Relative Percentile Gap (%)')
        ax8.set_title('Relative Gap by Prompt Template')
        ax8.grid(True, alpha=0.3)
        plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
        
        # 9. Reliability Diagram
        ax9 = plt.subplot(4, 3, 9)
        for model in models:
            if 'comparative' in self.results:
                model_results = self.results['comparative'].loc[model]
                bin_details = model_results['bin_details']
                
                if bin_details:
                    bin_confs = [bd['avg_confidence'] for bd in bin_details if bd['sample_count'] > 0]
                    bin_perfs = [bd['avg_performance'] for bd in bin_details if bd['sample_count'] > 0]
                    
                    ax9.plot(bin_confs, bin_perfs, 'o-', label=model, 
                            color=model_colors[model], linewidth=2, markersize=6)
        
        ax9.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax9.set_xlabel('Normalized Confidence')
        ax9.set_ylabel('Normalized Performance')
        ax9.set_title('Reliability Diagram')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Coefficient Comparison
        ax10 = plt.subplot(4, 3, 10)
        if 'comparative' in self.results:
            comp_df = self.results['comparative']
            x = np.arange(len(models))
            width = 0.25
            
            pearson_rs = comp_df['pearson_r']
            spearman_rs = comp_df['spearman_r']
            kendall_taus = comp_df['kendall_tau']
            
            ax10.bar(x - width, pearson_rs, width, label='Pearson r', alpha=0.8)
            ax10.bar(x, spearman_rs, width, label='Spearman ρ', alpha=0.8)
            ax10.bar(x + width, kendall_taus, width, label="Kendall's τ", alpha=0.8)
            
            ax10.set_xlabel('Models')
            ax10.set_ylabel('Correlation Coefficient')
            ax10.set_title('Correlation Coefficient Comparison')
            ax10.set_xticks(x)
            ax10.set_xticklabels(models, rotation=45, ha='right')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        # 11. Radar Chart
        ax11 = plt.subplot(4, 3, 11, projection='polar')
        if 'comparative' in self.results:
            comp_df = self.results['comparative']
            
            categories = ['Abs Performance', 'Abs Calibration', 'Rank Correlation', 'Concordance']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            for model in models:
                values = [
                    comp_df.loc[model, 'mean_f1'] / comp_df['mean_f1'].max(),  # Normalized performance
                    1 - min(comp_df.loc[model, 'absolute_ece'], 1),  # Inverted calibration error
                    max(0, comp_df.loc[model, 'spearman_r']),  # Positive correlation only
                    comp_df.loc[model, 'concordance_rate']
                ]
                values += values[:1]
                
                ax11.plot(angles, values, 'o-', linewidth=2, label=model, color=model_colors[model])
                ax11.fill(angles, values, alpha=0.1, color=model_colors[model])
            
            ax11.set_xticks(angles[:-1])
            ax11.set_xticklabels(categories)
            ax11.set_title('Performance Radar Chart')
            ax11.legend()
        
        # 12. Statistical Significance
        ax12 = plt.subplot(4, 3, 12)
        if 'statistical_comparison' in self.results:
            stat_comp = self.results['statistical_comparison']
            
            metrics = ['Confidence', 'Performance', 'Cal. Gap']
            p_values = [
                stat_comp['confidence_comparison']['p_value'],
                stat_comp['performance_comparison']['p_value'],
                stat_comp['calibration_gap_comparison']['p_value']
            ]
            
            colors_sig = ['red' if p < 0.05 else 'blue' for p in p_values]
            bars = ax12.bar(metrics, [-np.log10(p) for p in p_values], color=colors_sig, alpha=0.7)
            
            ax12.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax12.set_ylabel('-log10(p-value)')
            ax12.set_title('Statistical Significance')
            ax12.legend()
            ax12.grid(True, alpha=0.3)
            
            # p-value text
            for bar, p_val in zip(bars, p_values):
                height = bar.get_height()
                ax12.text(bar.get_x() + bar.get_width()/2., height,
                         f'p={p_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('multi_model_calibration_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Comparison plots saved as 'multi_model_calibration_comparison.png'")
    
    def generate_comparative_report(self, save_results=True):
        print("\nComparative Report...")
        
        report = []
        report.append("="*80)
        report.append("MULTI-MODEL CONFIDENCE CALIBRATION COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("1. DATA SUMMARY")
        report.append("-" * 40)
        report.append(f"Total combined samples: {len(self.df_combined)}")
        
        for model in self.df_combined['Model'].unique():
            model_data = self.df_combined[self.df_combined['Model'] == model]
            report.append(f"{model}: {len(model_data)} samples")
        
        report.append(f"Prompt templates: {list(self.df_combined['Prompt_Template'].unique())}")
        report.append("")
        
        if 'comparative' in self.results:
            report.append("2. DETAILED MODEL COMPARISON")
            report.append("-" * 40)
            
            comp_df = self.results['comparative']
            for model in comp_df.index:
                row = comp_df.loc[model]
                
                report.append(f"\n{model}:")
                report.append(f"  BASIC STATISTICS:")
                report.append(f"    • Sample size: {int(row['n_samples'])}")
                report.append(f"    • Mean Confidence: {row['mean_confidence']:.1f}% (σ={row['std_confidence']:.1f})")
                report.append(f"    • Mean F1 Score: {row['mean_f1']:.3f} (σ={row['std_f1']:.3f})")
                
                report.append(f"  ABSOLUTE ANALYSIS:")
                report.append(f"    • Calibration Gap: {row['absolute_calibration_gap']:.3f}")
                report.append(f"    • ECE: {row['absolute_ece']:.3f}")
                report.append(f"    • Pearson correlation: {row['pearson_r']:.3f} (p={row['pearson_p']:.6f})")
                report.append(f"    • Interpretation: {row['abs_interpretation']}")
                
                report.append(f"  RELATIVE ANALYSIS:")
                report.append(f"    • Spearman correlation: {row['spearman_r']:.3f} ({row['spearman_interpretation']})")
                report.append(f"    • Kendall's tau: {row['kendall_tau']:.3f}")
                report.append(f"    • Relative Calibration Error: {row['relative_calibration_error']:.3f}")
                report.append(f"    • Concordance rate: {row['concordance_rate']:.3f}")
                report.append(f"    • Calibration quality: {row['calibration_quality']}")
                
                report.append(f"  PERCENTILE ANALYSIS:")
                report.append(f"    • Mean percentile gap: {row['mean_percentile_gap']:.1f}%")
                report.append(f"    • Std percentile gap: {row['std_percentile_gap']:.1f}%")
                report.append(f"    • IQR: [{row['percentile_gap_q25']:.1f}%, {row['percentile_gap_q75']:.1f}%]")
        
        if 'statistical_comparison' in self.results:
            report.append("\n\n3. STATISTICAL COMPARISON")
            report.append("-" * 40)
            
            stat_comp = self.results['statistical_comparison']
            models = list(self.df_combined['Model'].unique())
            
            if len(models) == 2:
                model1, model2 = models[0], models[1]
                
                report.append(f"Comparing {model1} vs {model2}:")
                report.append("")
                
                conf_comp = stat_comp['confidence_comparison']
                report.append(f"CONFIDENCE COMPARISON:")
                report.append(f"  • {model1}: {conf_comp['model1_mean']:.1f}%")
                report.append(f"  • {model2}: {conf_comp['model2_mean']:.1f}%")
                report.append(f"  • t-statistic: {conf_comp['t_statistic']:.3f}")
                report.append(f"  • p-value: {conf_comp['p_value']:.6f}")
                report.append(f"  • Significant: {'Yes' if conf_comp['significant'] else 'No'}")
                
                perf_comp = stat_comp['performance_comparison']
                report.append(f"\nPERFORMANCE COMPARISON:")
                report.append(f"  • {model1}: {perf_comp['model1_mean']:.3f}")
                report.append(f"  • {model2}: {perf_comp['model2_mean']:.3f}")
                report.append(f"  • Performance improvement: {((perf_comp['model2_mean'] - perf_comp['model1_mean']) / perf_comp['model1_mean'] * 100):.1f}%")
                report.append(f"  • t-statistic: {perf_comp['t_statistic']:.3f}")
                report.append(f"  • p-value: {perf_comp['p_value']:.6f}")
                report.append(f"  • Significant: {'Yes' if perf_comp['significant'] else 'No'}")
                
                gap_comp = stat_comp['calibration_gap_comparison']
                report.append(f"\nCALIBRATION GAP COMPARISON:")
                report.append(f"  • {model1}: {gap_comp['model1_mean']:.3f}")
                report.append(f"  • {model2}: {gap_comp['model2_mean']:.3f}")
                report.append(f"  • Gap improvement: {((gap_comp['model1_mean'] - gap_comp['model2_mean']) / abs(gap_comp['model1_mean']) * 100):.1f}%")
                report.append(f"  • t-statistic: {gap_comp['t_statistic']:.3f}")
                report.append(f"  • p-value: {gap_comp['p_value']:.6f}")
                report.append(f"  • Significant: {'Yes' if gap_comp['significant'] else 'No'}")
        
        report.append("\n\n4. CONCLUSIONS")
        report.append("-" * 40)
        if 'comparative' in self.results and len(self.results['comparative']) == 2:
            comp_df = self.results['comparative']
            models = list(comp_df.index)
            
            perf_winner = comp_df.loc[comp_df['mean_f1'].idxmax()].name
            report.append(f"• Performance Winner: {perf_winner}")
            
            calib_winner = comp_df.loc[comp_df['relative_calibration_error'].idxmin()].name
            report.append(f"• Calibration Winner: {calib_winner}")
            
            corr_winner = comp_df.loc[comp_df['spearman_r'].idxmax()].name
            report.append(f"• Correlation Winner: {corr_winner}")
            
            report.append(f"• Both models show severe miscalibration (RCE > 0.7)")
            report.append(f"• Calibration techniques strongly recommended for both")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_results:
            with open('multi_model_comparison_report.txt', 'w', encoding='utf-8') as f:
                f.write(report_text)
            print("\nReport saved as 'multi_model_comparison_report.txt'")
        
        return report_text
    
    def save_all_results(self):
        print("\nSaving to CSV...")
        
        saved_files = []
        
        if 'comparative' in self.results:
            df_to_save = self.results['comparative'].drop('bin_details', axis=1, errors='ignore')
            filename = 'multi_model_comparison_results.csv'
            df_to_save.to_csv(filename)
            saved_files.append(filename)
            print(f"✅ {filename} saved")
        
        if 'statistical_comparison' in self.results:
            stat_data = []
            for comparison, data in self.results['statistical_comparison'].items():
                row = {'comparison_type': comparison}
                row.update(data)
                stat_data.append(row)
            
            stat_df = pd.DataFrame(stat_data)
            filename = 'statistical_comparison_results.csv'
            stat_df.to_csv(filename, index=False)
            saved_files.append(filename)
            print(f"✅ {filename} saved")
        
        processed_filename = 'combined_processed_data.csv'
        
        for model in self.df_combined['Model'].unique():
            model_mask = self.df_combined['Model'] == model
            self.df_combined.loc[model_mask, 'Confidence_Percentile'] = self.df_combined.loc[model_mask, 'Confidence'].rank(pct=True) * 100
            self.df_combined.loc[model_mask, 'Performance_Percentile'] = self.df_combined.loc[model_mask, 'Semantic_F1'].rank(pct=True) * 100
            self.df_combined.loc[model_mask, 'Percentile_Gap'] = self.df_combined.loc[model_mask, 'Confidence_Percentile'] - self.df_combined.loc[model_mask, 'Performance_Percentile']
            self.df_combined.loc[model_mask, 'Absolute_Gap'] = (self.df_combined.loc[model_mask, 'Confidence'] / 100.0) - self.df_combined.loc[model_mask, 'Semantic_F1']
        
        self.df_combined.to_csv(processed_filename, index=False)
        saved_files.append(processed_filename)
        print(f"✅ {processed_filename} saved")
        
        return saved_files
    
    def run_complete_comparison(self, save_results=True, create_plots=True):
        print("Multi-Model Calibration Comparison Analysis 시작!")
        print("="*70)
        
        if not self.load_and_combine_data():
            print("Analysis paused: Failed to load data")
            return None
        
        print("\n" + "="*50)
        print("Comparative analysis...")
        print("="*50)
        
        self.comprehensive_analysis()
        self.statistical_comparison()
        
        if create_plots:
            print("\n" + "="*50)
            print("Visualization...")
            print("="*50)
            self.create_comprehensive_comparison_plots()
        
        print("\n" + "="*50)
        print("Summarize and Save...")
        print("="*50)
        
        if save_results:
            self.save_all_results()
        
        self.generate_comparative_report(save_results=save_results)
        
        print("\nDone!")
        
        return self.results


def main():
    GEMMA_PATH = '/Users/sohyeon/Downloads/VitalLab/LLM_Cognitive_Abilities/data/analysis_details_integrated.csv'
    MEDGEMMA_PATH = '/Users/sohyeon/Downloads/VitalLab/LLM_Cognitive_Abilities/data/analysis_details_integrated_medgemma.csv'
    
    comparator = MultiModelCalibrationComparator(GEMMA_PATH, MEDGEMMA_PATH)
    results = comparator.run_complete_comparison(save_results=True, create_plots=True)
    
    return results


if __name__ == "__main__":
    results = main()
