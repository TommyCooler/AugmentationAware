"""
Main entry point for ACIIDS2025 time series anomaly detection project.

Phase 1: Contrastive Learning (Self-supervised)
Phase 2: Fine-tuning for Anomaly Detection (Supervised)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def phase1_train():
    """Run Phase 1 training"""
    from phase1.train_phase1 import main
    print("\n🚀 Starting Phase 1: Contrastive Learning Training...")
    main()


def phase2_train():
    """Run Phase 2 training (to be implemented)"""
    print("\n🚀 Starting Phase 2: Fine-tuning for Anomaly Detection...")
    print("⚠️  Phase 2 training not yet implemented.")


def main():
    parser = argparse.ArgumentParser(
        description='ACIIDS2025: Time Series Anomaly Detection with Contrastive Learning'
    )
    
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        default=1,
        help='Training phase: 1 (contrastive learning) or 2 (anomaly detection)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Run appropriate phase
    if args.phase == 1:
        phase1_train()
    elif args.phase == 2:
        phase2_train()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


