#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test scFoundation integration with real single-cell data.

This script demonstrates the full pipeline:
1. Load single-cell data from TDC
2. Tokenize with scFoundationTokenizer
3. Load model and extract cell embeddings

Usage:
    python scripts/test_scfoundation_real_data.py

    # With your own h5ad file:
    python scripts/test_scfoundation_real_data.py --data /path/to/your/data.h5ad

    # To download and use full pretrained weights:
    python scripts/test_scfoundation_real_data.py --use-pretrained
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def main(use_pretrained=False, data_path=None):
    print("=" * 60)
    print("scFoundation Integration Test with Real Data")
    print("=" * 60)

    # Step 1: Load single-cell data
    print("\n[1/4] Loading single-cell data...")

    adata = None

    # Try loading with scanpy (standard approach)
    try:
        import scanpy as sc
        # Check for h5ad file locations
        h5ad_paths = [
            data_path,  # User-provided path
            "./data/sample_data.h5ad",
            "./data/cellxgene_sample_small.h5ad",
            os.path.expanduser("~/data/sample_data.h5ad"),
        ]
        for path in h5ad_paths:
            if path is None:
                continue
            if os.path.exists(path):
                adata = sc.read_h5ad(path)
                print(f"  Loaded AnnData from {path}: {adata.shape[0]} cells x {adata.shape[1]} genes")
                break
    except ImportError:
        print("  scanpy not installed, will use synthetic data")
    except Exception as e:
        print(f"  Could not load h5ad: {e}")

    if adata is None:
        print("  Using synthetic test data instead...")

        # Create synthetic data for testing
        n_cells = 10
        n_genes = 500
        expression_matrix = np.random.rand(n_cells, n_genes).astype(np.float32) * 100
        expression_matrix[expression_matrix < 30] = 0  # Add sparsity

        # We'll get gene names from tokenizer
        from tdc_ml.model_server.tokenizers.scfoundation import scFoundationTokenizer
        temp_tokenizer = scFoundationTokenizer()
        gene_names = temp_tokenizer.gene_vocab[:n_genes]

        class FakeAdata:
            def __init__(self, X, var_names):
                self.X = X
                self.var_names = var_names
                self.shape = X.shape

        adata = FakeAdata(expression_matrix, gene_names)
        print(f"  Created synthetic data: {n_cells} cells x {n_genes} genes")

    # Step 2: Initialize tokenizer
    print("\n[2/4] Initializing tokenizer...")
    from tdc_ml.model_server.tokenizers.scfoundation import scFoundationTokenizer
    tokenizer = scFoundationTokenizer(data_path="./data")
    print(f"  Gene vocabulary size: {len(tokenizer.gene_vocab)}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  MASK token ID: {tokenizer.mask_token_id}")

    # Step 3: Tokenize the data
    print("\n[3/4] Tokenizing expression data...")

    # Get expression matrix (handle sparse matrices)
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = np.asarray(adata.X)

    gene_names = list(adata.var_names)

    tokens = tokenizer.tokenize_cell_vectors(
        expression_matrix,
        gene_names,
        target_sum=10000,
        pre_normalized='F',  # Raw counts
        return_tensors='pt',
        include_decoder=False  # Cell embeddings only
    )

    print(f"  Encoder data shape: {tokens['encoder_data'].shape}")
    print(f"  Position IDs shape: {tokens['encoder_position_gene_ids'].shape}")
    print(f"  Padding mask shape: {tokens['encoder_padding_mask'].shape}")
    print(f"  Non-padded genes per cell: {(~tokens['encoder_padding_mask']).sum(dim=1).tolist()}")

    # Step 4: Load model and get embeddings
    print("\n[4/4] Loading model and extracting embeddings...")

    if use_pretrained:
        print("  Loading pretrained model (this will download ~400MB on first run)...")
        from tdc_ml.model_server.tdc_hf import tdc_hf_interface
        model = tdc_hf_interface("scFoundation").load()
    else:
        print("  Using small test model (no pretrained weights)...")
        from tdc_ml.model_server.models.scfoundation import (
            scFoundationModel, scFoundationConfig
        )
        # Use smaller config for quick testing
        config = scFoundationConfig(
            encoder_depth=2,
            decoder_depth=2,
            encoder_hidden_dim=256,
            decoder_hidden_dim=128,
            encoder_heads=4,
            decoder_heads=4
        )
        model = scFoundationModel(config)

    model.eval()

    with torch.no_grad():
        # Get cell embeddings with different pooling strategies
        cell_emb_all = model.get_cell_embedding(
            tokens['encoder_data'],
            tokens['encoder_position_gene_ids'],
            tokens['encoder_padding_mask'],
            pool_type='all'
        )

        cell_emb_mean = model.get_cell_embedding(
            tokens['encoder_data'],
            tokens['encoder_position_gene_ids'],
            tokens['encoder_padding_mask'],
            pool_type='mean'
        )

    print(f"\n  Cell embeddings (all pooling): {cell_emb_all.shape}")
    print(f"  Cell embeddings (mean pooling): {cell_emb_mean.shape}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Input cells: {expression_matrix.shape[0]}")
    print(f"  Input genes: {expression_matrix.shape[1]}")
    print(f"  Aligned to vocabulary: {tokenizer.NUM_GENES} genes")
    print(f"  Embedding dim (all): {cell_emb_all.shape[1]}")
    print(f"  Embedding dim (mean): {cell_emb_mean.shape[1]}")
    print(f"\n  Embedding statistics (all pooling):")
    print(f"    Mean: {cell_emb_all.mean().item():.4f}")
    print(f"    Std:  {cell_emb_all.std().item():.4f}")
    print(f"    Min:  {cell_emb_all.min().item():.4f}")
    print(f"    Max:  {cell_emb_all.max().item():.4f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return cell_emb_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test scFoundation with real data")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Download and use full pretrained weights (~400MB)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to h5ad file with single-cell data"
    )
    args = parser.parse_args()

    main(use_pretrained=args.use_pretrained, data_path=args.data)
