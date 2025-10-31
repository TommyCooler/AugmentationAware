"""
Visualization utilities for Phase 2 Inference
Visualize original data, augmented data, reconstruction, and anomaly regions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple


def visualize_inference_results(
    original_data: np.ndarray,
    augmented_data: np.ndarray,
    reconstructed_data: np.ndarray,
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    channels_to_plot: Optional[list] = None,
    max_length: int = 2000,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize inference results: original data, augmented, reconstruction, and anomaly regions
    
    Args:
        original_data: Original test data, shape (n_channels, n_time_steps) or (n_time_steps, n_channels)
        augmented_data: Augmented data, same shape as original_data
        reconstructed_data: Reconstructed data, same shape as original_data
        anomaly_scores: Anomaly scores per time step, shape (n_time_steps,)
        labels: Ground truth labels, shape (n_time_steps,)
        predictions: Predicted labels (optional), shape (n_time_steps,)
        threshold: Threshold used for predictions (optional)
        save_path: Path to save figure (optional)
        channels_to_plot: List of channel indices to plot (None = plot all)
        max_length: Maximum time steps to plot (for very long sequences)
        figsize: Figure size (width, height)
    """
    # Handle shape: convert (n_time_steps, n_channels) to (n_channels, n_time_steps)
    if original_data.shape[0] != anomaly_scores.shape[0]:
        original_data = original_data.T
        augmented_data = augmented_data.T
        reconstructed_data = reconstructed_data.T
    
    n_channels, n_time_steps = original_data.shape
    
    # Limit length if too long
    if n_time_steps > max_length:
        print(f"⚠️  Limiting visualization to first {max_length} time steps")
        original_data = original_data[:, :max_length]
        augmented_data = augmented_data[:, :max_length]
        reconstructed_data = reconstructed_data[:, :max_length]
        anomaly_scores = anomaly_scores[:max_length]
        labels = labels[:max_length]
        if predictions is not None:
            predictions = predictions[:max_length]
        n_time_steps = max_length
    
    # Select channels to plot
    if channels_to_plot is None:
        channels_to_plot = list(range(n_channels))
    
    # Create time axis
    time_axis = np.arange(n_time_steps)
    
    # Determine number of subplots: 1 for scores + n_channels for data
    n_plots = len(channels_to_plot) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot anomaly scores at the top
    ax = axes[0]
    
    # Background: Ground truth anomalies
    for i in range(n_time_steps):
        if labels[i] > 0:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red', label='Ground Truth' if i == 0 else '')
    
    # Background: Predicted anomalies
    if predictions is not None:
        for i in range(n_time_steps):
            if predictions[i] > 0:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='orange', label='Prediction' if i == 0 and np.any(predictions > 0) else '')
    
    # Plot anomaly scores
    ax.plot(time_axis, anomaly_scores, 'b-', linewidth=1.5, label='Anomaly Score', alpha=0.7)
    
    # Plot threshold line
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    ax.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Plot each channel
    for idx, channel_idx in enumerate(channels_to_plot):
        ax = axes[idx + 1]
        
        # Background: Ground truth anomalies
        for i in range(n_time_steps):
            if labels[i] > 0:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red', zorder=0)
        
        # Background: Predicted anomalies
        if predictions is not None:
            for i in range(n_time_steps):
                if predictions[i] > 0:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='orange', zorder=0)
        
        # Plot original data
        ax.plot(time_axis, original_data[channel_idx], 'k-', linewidth=1.5, label='Original', alpha=0.7)
        
        # Plot augmented data
        ax.plot(time_axis, augmented_data[channel_idx], 'g--', linewidth=1.2, label='Augmented', alpha=0.6, dashes=(3, 2))
        
        # Plot reconstructed data
        ax.plot(time_axis, reconstructed_data[channel_idx], 'b-', linewidth=1.2, label='Reconstructed', alpha=0.6)
        
        # Compute and plot reconstruction error
        recon_error = np.abs(reconstructed_data[channel_idx] - augmented_data[channel_idx])
        ax_twin = ax.twinx()
        ax_twin.fill_between(time_axis, 0, recon_error, alpha=0.2, color='purple', label='Recon Error')
        ax_twin.set_ylabel('Reconstruction Error', fontsize=10, color='purple')
        ax_twin.tick_params(axis='y', labelcolor='purple')
        
        ax.set_ylabel(f'Channel {channel_idx}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax_twin.legend(loc='upper right', fontsize=9)
    
    # Set x-axis label on last subplot
    axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.show()


def visualize_summary(
    original_data: np.ndarray,
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    max_length: int = 2000,
    figsize: Tuple[int, int] = (16, 8)
):
    """
    Simplified visualization: only anomaly scores and regions
    
    Args:
        original_data: Original test data, shape (n_channels, n_time_steps) or (n_time_steps, n_channels)
        anomaly_scores: Anomaly scores per time step, shape (n_time_steps,)
        labels: Ground truth labels, shape (n_time_steps,)
        predictions: Predicted labels (optional), shape (n_time_steps,)
        threshold: Threshold used for predictions (optional)
        save_path: Path to save figure (optional)
        max_length: Maximum time steps to plot
        figsize: Figure size (width, height)
    """
    # Handle shape
    if original_data.shape[0] != anomaly_scores.shape[0]:
        original_data = original_data.T
    
    n_channels, n_time_steps = original_data.shape
    
    # Limit length
    if n_time_steps > max_length:
        print(f"⚠️  Limiting visualization to first {max_length} time steps")
        original_data = original_data[:, :max_length]
        anomaly_scores = anomaly_scores[:max_length]
        labels = labels[:max_length]
        if predictions is not None:
            predictions = predictions[:max_length]
        n_time_steps = max_length
    
    time_axis = np.arange(n_time_steps)
    
    fig, axes = plt.subplots(n_channels + 1, 1, figsize=figsize, sharex=True)
    
    if n_channels == 0:
        axes = [axes]
    
    # Plot anomaly scores
    ax = axes[0]
    
    # Background: Ground truth anomalies
    for i in range(n_time_steps):
        if labels[i] > 0:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red', label='Ground Truth' if i == 0 else '')
    
    # Background: Predicted anomalies
    if predictions is not None:
        for i in range(n_time_steps):
            if predictions[i] > 0:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='orange', label='Prediction' if i == 0 and np.any(predictions > 0) else '')
    
    ax.plot(time_axis, anomaly_scores, 'b-', linewidth=1.5, label='Anomaly Score', alpha=0.7)
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    ax.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_title('Anomaly Detection Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Plot each channel
    for channel_idx in range(n_channels):
        ax = axes[channel_idx + 1]
        
        # Background: Ground truth anomalies
        for i in range(n_time_steps):
            if labels[i] > 0:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red', zorder=0)
        
        # Background: Predicted anomalies
        if predictions is not None:
            for i in range(n_time_steps):
                if predictions[i] > 0:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='orange', zorder=0)
        
        ax.plot(time_axis, original_data[channel_idx], 'k-', linewidth=1.5, label=f'Channel {channel_idx}', alpha=0.7)
        ax.set_ylabel(f'Channel {channel_idx}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    
    axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.show()

