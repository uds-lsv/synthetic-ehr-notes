import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import argparse

def plot_f1_vs_frequency_compare(output_file, f1_df, f1_df2=None, num_bins=500):
    # plot f1 against training code frequency comparing the real and synthetic-data models (or optionally just plotting for one model)
    f1_df['frequency_bin'] = pd.qcut(f1_df['frequency'], num_bins, duplicates='drop')
    grouped = f1_df.groupby('frequency_bin').agg(
        mean_f1=('f1', 'mean'),
        std_f1=('f1', 'std'),
        count=('f1', 'count')
    ).reset_index()
    grouped['bin_mid'] = grouped['frequency_bin'].apply(lambda x: x.mid)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='bin_mid', y='mean_f1', data=grouped, label='Average F1 Score (Real)', color='blue')
    plt.fill_between(grouped['bin_mid'], grouped['mean_f1'] - grouped['std_f1'], 
                     grouped['mean_f1'] + grouped['std_f1'], color='blue', alpha=0.3, label='±1 Std Dev (Real)')

    if f1_df2 is not None:
        f1_df2['frequency_bin'] = pd.qcut(f1_df2['frequency'], num_bins, duplicates='drop')
        grouped2 = f1_df2.groupby('frequency_bin').agg(
            mean_f1=('f1', 'mean'),
            std_f1=('f1', 'std'),
            count=('f1', 'count')
        ).reset_index()
        grouped2['bin_mid'] = grouped2['frequency_bin'].apply(lambda x: x.mid)
        sns.lineplot(x='bin_mid', y='mean_f1', data=grouped2, label='Average F1 Score (Synth)', color='green')
        plt.fill_between(grouped2['bin_mid'], grouped2['mean_f1'] - grouped2['std_f1'], 
                         grouped2['mean_f1'] + grouped2['std_f1'], color='green', alpha=0.3, label='±1 Std Dev (Synth)')

    plt.xscale('log')
    plt.title('Average F1 Score vs Frequency', fontsize=14)
    plt.xlabel('Frequency (Log Scale)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

def correlation_coefficients(f1_df, label):
    # calculate correlation coefficients for f1 against code frequency
    f1_df = f1_df[f1_df['frequency'] > 0]
    f1_df['log_frequency'] = np.log(f1_df['frequency'])

    pearson_corr, pearson_p_value = pearsonr(f1_df['f1'], f1_df['log_frequency'])
    spearman_corr, spearman_p_value = spearmanr(f1_df['f1'], f1_df['log_frequency'])

    print(f"\nCorrelations for {label}:")
    print(f"  Pearson Correlation: {pearson_corr:.3f}")
    print(f"  Pearson p-value: {pearson_p_value:.3g}")
    print(f"  Spearman Correlation: {spearman_corr:.3f}")
    print(f"  Spearman p-value: {spearman_p_value:.3g}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_f1_frequ", required=True, help="Path to the CSV file for real data.")
    parser.add_argument("--synth_f1_frequ", help="Path to the CSV file for synthetic data (optional).")
    parser.add_argument("--plot_output", required=True, help="Path to save the plot.")
    parser.add_argument("--num_bins", type=int, default=500, help="Number of bins for frequency binning.")
    
    args = parser.parse_args()

    # Load datasets
    real = pd.read_csv(args.real_f1_frequ)
    synth = pd.read_csv(args.synth_f1_frequ) if args.synth_path else None

    # Plot F1 scores vs frequency
    plot_f1_vs_frequency_compare(args.plot_output, real, synth, num_bins=args.num_bins)

    # Calculate correlations
    real_corr = correlation_coefficients(real)
    synth_corr = correlation_coefficients(synth) if synth is not None else None
