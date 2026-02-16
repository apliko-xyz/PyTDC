# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
# Adapted for PyTDC integration

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union

from ...utils.load import download_wrapper


class scFoundationTokenizer:
    """
    Tokenizer for scFoundation single-cell foundation model.

    Handles gene alignment to the model's 19,264 gene vocabulary,
    normalization (log1p(CPM)), and sparse gene gathering for
    encoder/decoder input preparation.

    Reference: https://github.com/biomap-research/scFoundation
    """

    NUM_GENES = 19264
    PAD_TOKEN_ID = 19264
    MASK_TOKEN_ID = 19265
    SEQ_LEN = 19264

    def __init__(self, gene_vocab_path: Optional[str] = None, data_path: str = "./data"):
        """
        Initialize the scFoundation tokenizer.

        Args:
            gene_vocab_path: Path to gene vocabulary TSV file. If None, downloads from TDC.
            data_path: Directory for downloading/caching data files.
        """
        self.data_path = data_path
        self.gene_vocab = self._load_gene_vocab(gene_vocab_path)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_vocab)}
        self.pad_token_id = self.PAD_TOKEN_ID
        self.mask_token_id = self.MASK_TOKEN_ID
        self.seq_len = self.SEQ_LEN

    def _load_gene_vocab(self, gene_vocab_path: Optional[str] = None) -> List[str]:
        """
        Load the 19,264 gene vocabulary.

        Args:
            gene_vocab_path: Path to gene vocabulary TSV file.

        Returns:
            List of gene names in order.
        """
        if gene_vocab_path is not None and os.path.exists(gene_vocab_path):
            gene_df = pd.read_csv(gene_vocab_path, header=0, delimiter='\t')
            return list(gene_df['gene_name'])

        # Try to download from TDC storage
        try:
            download_wrapper("scfoundation_gene_vocab", self.data_path,
                           ["scfoundation_gene_vocab"])
            vocab_path = os.path.join(self.data_path, "scfoundation_gene_vocab.tsv")
            if os.path.exists(vocab_path):
                gene_df = pd.read_csv(vocab_path, header=0, delimiter='\t')
                return list(gene_df['gene_name'])
        except Exception:
            pass

        # Fallback: try local scFoundation path
        local_path = os.path.expanduser("~/scFoundation/model/OS_scRNA_gene_index.19264.tsv")
        if os.path.exists(local_path):
            gene_df = pd.read_csv(local_path, header=0, delimiter='\t')
            return list(gene_df['gene_name'])

        # Use loader to download gene vocabulary
        from model_server.model_loaders.scfoundation_loader import scFoundationLoader
        loader = scFoundationLoader()
        vocab_path = loader.load_gene_vocab(self.data_path)
        gene_df = pd.read_csv(vocab_path, header=0, delimiter='\t')
        return list(gene_df['gene_name'])

    def _align_genes(
        self,
        expression_matrix: np.ndarray,
        gene_names: Union[List[str], np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Align input genes to model's 19,264 gene vocabulary.

        Reorders columns to match vocabulary order and pads missing genes with zeros.

        Args:
            expression_matrix: Expression matrix of shape (n_cells, n_genes).
            gene_names: List of gene names corresponding to columns.

        Returns:
            Tuple of (aligned_matrix, missing_genes) where aligned_matrix has shape
            (n_cells, 19264) and missing_genes is a list of genes not in input.
        """
        gene_names = list(gene_names)
        n_cells = expression_matrix.shape[0]

        # Create aligned matrix
        aligned_matrix = np.zeros((n_cells, self.NUM_GENES), dtype=np.float32)

        # Create mapping from input gene names to their indices
        input_gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        # Fill in values for genes that exist in both
        missing_genes = []
        for vocab_idx, vocab_gene in enumerate(self.gene_vocab):
            if vocab_gene in input_gene_to_idx:
                input_idx = input_gene_to_idx[vocab_gene]
                aligned_matrix[:, vocab_idx] = expression_matrix[:, input_idx]
            else:
                missing_genes.append(vocab_gene)

        return aligned_matrix, missing_genes

    def _normalize_expression(
        self,
        expression: np.ndarray,
        target_sum: int = 10000,
        pre_normalized: str = 'F'
    ) -> np.ndarray:
        """
        Apply log1p(CPM) normalization to expression values.

        Args:
            expression: Expression matrix of shape (n_cells, n_genes).
            target_sum: Target sum for normalization (default 10000 for CPM).
            pre_normalized: 'F' if not normalized, 'T' if already normalized+log1p,
                          'A' if normalized+log1p with total count appended.

        Returns:
            Normalized expression matrix.
        """
        if pre_normalized == 'T':
            # Already normalized
            return expression.astype(np.float32)
        elif pre_normalized == 'A':
            # Already normalized, total count is appended (remove it)
            return expression[:, :-1].astype(np.float32)
        else:
            # Apply log1p(CPM) normalization: log1p(expr / sum * target_sum)
            row_sums = expression.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            normalized = np.log1p(expression / row_sums * target_sum)
            return normalized.astype(np.float32)

    def _gather_expressed_genes(
        self,
        expression: torch.Tensor,
        pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather non-zero (expressed) genes for sparse representation.

        This creates the encoder input by selecting only expressed genes,
        reducing sequence length from 19,264 to the number of expressed genes.

        Args:
            expression: Expression tensor of shape (batch, n_genes).
            pad_token_id: Token ID to use for padding.

        Returns:
            Tuple of (gathered_data, padding_mask, position_gene_ids).
        """
        device = expression.device
        batch_size = expression.shape[0]

        # Find expressed genes (non-zero values)
        value_labels = expression > 0
        value_nums = value_labels.sum(dim=1)
        max_num = max(value_nums).item()

        if max_num == 0:
            # No expressed genes - return minimal padding
            max_num = 1

        # Create padded data tensor
        fake_data = torch.full((batch_size, max_num), pad_token_id,
                               device=device, dtype=expression.dtype)
        padded_data = torch.hstack([expression, fake_data])

        # Create label tensor for sorting
        fake_label = torch.full((batch_size, max_num), 1.0, device=device)
        none_labels = ~value_labels
        labels = value_labels.float()
        labels[none_labels] = float('-inf')

        # Add position-based offset for stable sorting
        tmp_data = torch.tensor(
            [(i + 1) * 20000 for i in range(value_labels.shape[1], 0, -1)],
            device=device, dtype=labels.dtype
        )
        labels = labels + tmp_data
        labels = torch.hstack([labels, fake_label])

        # Get top-k indices (expressed genes first)
        topk_indices = labels.topk(max_num).indices

        # Gather data
        gathered_data = torch.gather(padded_data, 1, topk_indices)

        # Create padding mask
        padding_mask = (gathered_data == pad_token_id)

        # Create position gene IDs
        gene_ids = torch.arange(expression.shape[1], device=device).repeat(batch_size, 1)
        fake_gene_ids = torch.full((batch_size, max_num), self.seq_len, device=device)
        padded_gene_ids = torch.hstack([gene_ids, fake_gene_ids])
        position_gene_ids = torch.gather(padded_gene_ids, 1, topk_indices)

        # Set padding positions to seq_len
        position_gene_ids[padding_mask] = self.seq_len

        return gathered_data, padding_mask, position_gene_ids

    def _prepare_encoder_decoder_data(
        self,
        expression: torch.Tensor,
        raw_expression: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for both encoder and decoder.

        Args:
            expression: Normalized expression tensor (batch, n_genes).
            raw_expression: Raw expression tensor for determining expressed genes.

        Returns:
            Dictionary containing encoder and decoder tensors.
        """
        device = expression.device
        batch_size = expression.shape[0]

        # Decoder uses full sequence
        decoder_data = expression.clone()
        decoder_padding_mask = torch.zeros(
            batch_size, self.NUM_GENES, dtype=torch.bool, device=device
        )

        # Encoder uses gathered (sparse) data
        encoder_labels = raw_expression > 0
        encoder_data, encoder_padding_mask, encoder_position_gene_ids = \
            self._gather_expressed_genes(decoder_data, self.pad_token_id)

        # Decoder position gene IDs
        decoder_position_gene_ids = torch.arange(
            self.NUM_GENES, device=device
        ).repeat(batch_size, 1)

        return {
            'encoder_data': encoder_data,
            'encoder_position_gene_ids': encoder_position_gene_ids,
            'encoder_padding_mask': encoder_padding_mask,
            'encoder_labels': encoder_labels,
            'decoder_data': decoder_data,
            'decoder_position_gene_ids': decoder_position_gene_ids,
            'decoder_padding_mask': decoder_padding_mask,
        }

    def tokenize_cell_vectors(
        self,
        expression_matrix: Union[np.ndarray, "scipy.sparse.spmatrix"],
        gene_names: Union[List[str], np.ndarray],
        target_sum: int = 10000,
        pre_normalized: str = 'F',
        return_tensors: str = 'pt',
        include_decoder: bool = False,
        tgt_high_res: str = 't4'
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Tokenize single-cell expression data for scFoundation.

        This is the main entry point for tokenization. It:
        1. Aligns input genes to the 19,264 gene vocabulary
        2. Normalizes expression values (log1p(CPM))
        3. Gathers expressed genes for encoder input
        4. Optionally prepares decoder input for gene embedding extraction

        Args:
            expression_matrix: Expression matrix of shape (n_cells, n_genes).
                Can be dense numpy array or scipy sparse matrix.
            gene_names: List of gene names corresponding to columns.
            target_sum: Target sum for CPM normalization (default 10000).
            pre_normalized: 'F' if raw counts, 'T' if already log1p(CPM),
                          'A' if normalized with total count appended.
            return_tensors: 'pt' for PyTorch tensors, 'np' for numpy arrays.
            include_decoder: If True, include decoder data for gene embeddings.
            tgt_high_res: Target high resolution for cell embedding.
                         Format: 't{value}' for target, 'f{factor}' for fold change,
                         'a{addition}' for addition.

        Returns:
            Dictionary containing:
                - 'encoder_data': Gathered expression values (batch, max_expressed)
                - 'encoder_position_gene_ids': Gene indices (batch, max_expressed)
                - 'encoder_padding_mask': Padding mask (batch, max_expressed)
            If include_decoder=True, also includes:
                - 'decoder_data': Full expression (batch, 19264)
                - 'decoder_position_gene_ids': Gene indices (batch, 19264)
                - 'decoder_padding_mask': Padding mask (batch, 19264)
                - 'encoder_labels': Which genes are expressed (batch, 19264)
        """
        # Convert sparse matrix to dense if needed
        if hasattr(expression_matrix, 'toarray'):
            expression_matrix = expression_matrix.toarray()

        expression_matrix = np.asarray(expression_matrix, dtype=np.float32)

        # Align genes to vocabulary
        aligned_matrix, _ = self._align_genes(expression_matrix, gene_names)

        # Normalize expression
        normalized_matrix = self._normalize_expression(
            aligned_matrix, target_sum, pre_normalized
        )

        # Add resolution tokens (last 2 positions)
        # Position 19264: target resolution, Position 19265: current resolution
        batch_size = normalized_matrix.shape[0]

        # Calculate total counts for resolution
        total_counts = aligned_matrix.sum(axis=1)
        current_res = np.log10(np.where(total_counts > 0, total_counts, 1))

        # Parse target resolution
        if tgt_high_res.startswith('t'):
            target_res = np.full(batch_size, float(tgt_high_res[1:]))
        elif tgt_high_res.startswith('f'):
            target_res = current_res + np.log10(float(tgt_high_res[1:]))
        elif tgt_high_res.startswith('a'):
            target_res = current_res + float(tgt_high_res[1:])
        else:
            target_res = np.full(batch_size, 4.0)  # Default

        # Append resolution tokens
        resolution_tokens = np.stack([target_res, current_res], axis=1).astype(np.float32)
        full_expression = np.concatenate([normalized_matrix, resolution_tokens], axis=1)

        # Convert to tensor
        expression_tensor = torch.from_numpy(full_expression)

        if include_decoder:
            # Prepare both encoder and decoder data
            result = self._prepare_encoder_decoder_data(
                expression_tensor[:, :self.NUM_GENES],
                expression_tensor[:, :self.NUM_GENES]
            )
            # Add resolution tokens back to decoder data
            result['decoder_data'] = expression_tensor
            result['decoder_position_gene_ids'] = torch.arange(
                self.NUM_GENES + 2
            ).repeat(batch_size, 1)
            result['decoder_padding_mask'] = torch.zeros(
                batch_size, self.NUM_GENES + 2, dtype=torch.bool
            )
        else:
            # Encoder only (for cell embeddings)
            encoder_data, encoder_padding_mask, encoder_position_gene_ids = \
                self._gather_expressed_genes(expression_tensor, self.pad_token_id)

            result = {
                'encoder_data': encoder_data,
                'encoder_position_gene_ids': encoder_position_gene_ids,
                'encoder_padding_mask': encoder_padding_mask,
            }

        if return_tensors == 'np':
            result = {k: v.numpy() for k, v in result.items()}

        return result

    def get_config(self) -> Dict:
        """
        Get tokenizer configuration for model initialization.

        Returns:
            Dictionary with model configuration parameters.
        """
        return {
            'pad_token_id': self.pad_token_id,
            'mask_token_id': self.mask_token_id,
            'seq_len': self.seq_len,
            'n_class': self.NUM_GENES,
            'num_genes': self.NUM_GENES,
        }
