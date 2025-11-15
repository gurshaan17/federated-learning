"""
Evaluation Script - Analyze training results
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics(metrics_path='./results/metrics.csv'):
    """Load metrics CSV file"""
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return None
    
    df = pd.read_csv(metrics_path)
    return df


def print_statistics(df):
    """Print training statistics"""
    print("\n" + "="*80)
    print("TRAINING STATISTICS")
    print("="*80)
    
    print(f"\nTotal Communication Rounds: {df['communication_round'].max() + 1}")
    print(f"Total Metrics Recorded: {len(df)}")
    
    print("\nGenerator Loss Statistics:")
    print(f"  Min: {df['generator_loss'].min():.6f}")
    print(f"  Max: {df['generator_loss'].max():.6f}")
    print(f"  Mean: {df['generator_loss'].mean():.6f}")
    print(f"  Std: {df['generator_loss'].std():.6f}")
    
    print("\nDiscriminator Loss Statistics:")
    print(f"  Min: {df['discriminator_loss'].min():.6f}")
    print(f"  Max: {df['discriminator_loss'].max():.6f}")
    print(f"  Mean: {df['discriminator_loss'].mean():.6f}")
    print(f"  Std: {df['discriminator_loss'].std():.6f}")
    
    print("\nDiscriminator Accuracy Statistics:")
    print(f"  Min: {df['discriminator_acc'].min():.6f}")
    print(f"  Max: {df['discriminator_acc'].max():.6f}")
    print(f"  Mean: {df['discriminator_acc'].mean():.6f}")
    print(f"  Std: {df['discriminator_acc'].std():.6f}")
    
    # Per-client statistics
    print("\nPer-Client Statistics:")
    for client_id in sorted(df['client_id'].unique()):
        client_df = df[df['client_id'] == client_id]
        print(f"\n  Client {int(client_id) + 1}:")
        print(f"    Avg Gen Loss: {client_df['generator_loss'].mean():.6f}")
        print(f"    Avg Disc Loss: {client_df['discriminator_loss'].mean():.6f}")
        print(f"    Avg Disc Acc: {client_df['discriminator_acc'].mean():.6f}")


def generate_report(df, output_path='./results/evaluation_report.txt'):
    """Generate detailed evaluation report"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEDERATED LEARNING WITH GANs - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Communication Rounds: {df['communication_round'].max() + 1}\n")
        f.write(f"Number of Clients: {df['client_id'].nunique()}\n")
        f.write(f"Total Metrics Recorded: {len(df)}\n\n")
        
        f.write("GENERATOR LOSS\n")
        f.write("-"*80 + "\n")
        f.write(f"Initial (Round 1 avg): {df[df['communication_round'] == 0]['generator_loss'].mean():.6f}\n")
        f.write(f"Final (Last round avg): {df[df['communication_round'] == df['communication_round'].max()]['generator_loss'].mean():.6f}\n")
        f.write(f"Improvement: {(df[df['communication_round'] == 0]['generator_loss'].mean() - df[df['communication_round'] == df['communication_round'].max()]['generator_loss'].mean()):.6f}\n\n")
        
        f.write("DISCRIMINATOR LOSS\n")
        f.write("-"*80 + "\n")
        f.write(f"Initial (Round 1 avg): {df[df['communication_round'] == 0]['discriminator_loss'].mean():.6f}\n")
        f.write(f"Final (Last round avg): {df[df['communication_round'] == df['communication_round'].max()]['discriminator_loss'].mean():.6f}\n")
        f.write(f"Change: {(df[df['communication_round'] == df['communication_round'].max()]['discriminator_loss'].mean() - df[df['communication_round'] == 0]['discriminator_loss'].mean()):.6f}\n\n")
        
        f.write("DISCRIMINATOR ACCURACY\n")
        f.write("-"*80 + "\n")
        f.write(f"Initial (Round 1 avg): {df[df['communication_round'] == 0]['discriminator_acc'].mean():.6f}\n")
        f.write(f"Final (Last round avg): {df[df['communication_round'] == df['communication_round'].max()]['discriminator_acc'].mean():.6f}\n\n")
        
        f.write("PER-CLIENT ANALYSIS\n")
        f.write("-"*80 + "\n")
        for client_id in sorted(df['client_id'].unique()):
            client_df = df[df['client_id'] == client_id]
            f.write(f"\nClient {int(client_id) + 1}:\n")
            f.write(f"  Total Rounds Participated: {client_df['communication_round'].nunique()}\n")
            f.write(f"  Avg Gen Loss: {client_df['generator_loss'].mean():.6f}\n")
            f.write(f"  Avg Disc Loss: {client_df['discriminator_loss'].mean():.6f}\n")
            f.write(f"  Avg Disc Acc: {client_df['discriminator_acc'].mean():.6f}\n")
    
    print(f"\nEvaluation report saved to: {output_path}")


def main():
    print("="*80)
    print("Federated Learning Training Evaluation")
    print("="*80)
    
    # Load metrics
    print("\nLoading metrics...")
    df = load_metrics()
    
    if df is None:
        return
    
    # Print statistics
    print_statistics(df)
    
    # Generate report
    generate_report(df)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
