"""
Visualization tools for Phase 1 training.
Helps understand embeddings, augmentations, and training progress.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def visualize_embeddings(encoder, dataloader, device='cuda', max_samples=1000):
    """
    Visualize learned embeddings using t-SNE or PCA.
    
    Args:
        encoder: Trained encoder model
        dataloader: DataLoader with windows
        device: Device to run on
        max_samples: Maximum number of samples to visualize
    """
    encoder.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Get original windows
            
            batch = batch.to(device)
            z = encoder(batch)
            embeddings.append(z.cpu().numpy())
            
            if len(embeddings) * batch.shape[0] >= max_samples:
                break
    
    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    
    # Apply t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE plot
    axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   alpha=0.5, s=10, c=np.arange(len(embeddings_2d)), cmap='viridis')
    axes[0].set_title('t-SNE Visualization of Embeddings')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # PCA plot
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    axes[1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                   alpha=0.5, s=10, c=np.arange(len(embeddings_pca)), cmap='viridis')
    axes[1].set_title('PCA Visualization of Embeddings')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    plt.tight_layout()
    plt.savefig('phase1/embeddings_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved embeddings visualization to phase1/embeddings_visualization.png")
    plt.close()


def visualize_augmentation(augmentation, dataloader, device='cuda', num_samples=5):
    """
    Visualize augmentation effects on sample windows.
    
    Args:
        augmentation: Augmentation module
        dataloader: DataLoader with windows
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    augmentation.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    
    batch = batch[:num_samples].to(device)
    
    with torch.no_grad():
        augmented = augmentation(batch)
    
    # Move to CPU for plotting
    original = batch.cpu().numpy()
    augmented = augmented.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 2*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original
        for ch in range(original.shape[1]):
            axes[i, 0].plot(original[i, ch], label=f'Channel {ch}', alpha=0.7)
        axes[i, 0].set_title(f'Original Window {i+1}')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Value')
        if original.shape[1] > 1:
            axes[i, 0].legend()
        
        # Augmented
        for ch in range(augmented.shape[1]):
            axes[i, 1].plot(augmented[i, ch], label=f'Channel {ch}', alpha=0.7)
        axes[i, 1].set_title(f'Augmented Window {i+1}')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Value')
        if augmented.shape[1] > 1:
            axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('phase1/augmentation_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved augmentation visualization to phase1/augmentation_visualization.png")
    plt.close()


def plot_training_curves(checkpoint_dir='phase1/checkpoints', loss_history=None):
    """
    Plot training loss curves.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        loss_history: List of (epoch, loss) tuples
    """
    if loss_history is None:
        print("⚠️  No loss history provided. Use wandb or save losses during training.")
        return
    
    epochs, losses = zip(*loss_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Phase 1 Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add min loss annotation
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    min_epoch = epochs[min_loss_idx]
    plt.annotate(f'Min: {min_loss:.4f}',
                xy=(min_epoch, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('phase1/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training curves to phase1/training_curves.png")
    plt.close()


def compute_similarity_matrix(encoder, dataloader, device='cuda', num_samples=100):
    """
    Compute and visualize similarity matrix between embeddings.
    
    Args:
        encoder: Trained encoder
        dataloader: DataLoader with windows
        device: Device to run on
        num_samples: Number of samples to compare
    """
    encoder.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(device)
            z = encoder(batch)
            embeddings.append(z)
            
            if len(embeddings) * batch.shape[0] >= num_samples:
                break
    
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    
    # Compute cosine similarity matrix
    similarity = torch.mm(embeddings, embeddings.t())
    similarity = similarity.cpu().numpy()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Embedding Similarity Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig('phase1/similarity_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved similarity matrix to phase1/similarity_matrix.png")
    plt.close()


if __name__ == '__main__':
    print("Visualization tools for Phase 1 training")
    print("\nAvailable functions:")
    print("  - visualize_embeddings(encoder, dataloader)")
    print("  - visualize_augmentation(augmentation, dataloader)")
    print("  - plot_training_curves(loss_history)")
    print("  - compute_similarity_matrix(encoder, dataloader)")
    print("\nExample usage:")
    print("""
    from phase1.train_phase1 import Phase1Trainer
    from phase1.encoder import MLPEncoder
    from phase1.visualize_training import visualize_embeddings
    
    # Load trained model
    encoder = MLPEncoder(...)
    checkpoint = torch.load('phase1/checkpoints/best_model.pth')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Visualize
    visualize_embeddings(encoder, dataloader)
    """)


