# -*- coding: utf-8 -*-
"""
Tests for scFoundation model integration in PyTDC.

This module contains unit tests for:
- scFoundationTokenizer: Gene vocab loading, alignment, normalization, tokenization
- scFoundationModel: Model architecture, forward pass, embedding extraction
- Integration: End-to-end pipeline testing
"""

import os
import sys
import unittest

import numpy as np
import torch

# Temporary solution for relative imports in case TDC is not installed
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestScFoundationTokenizer(unittest.TestCase):
    """Unit tests for scFoundationTokenizer."""

    @classmethod
    def setUpClass(cls):
        """Set up tokenizer once for all tests."""
        from tdc_ml.model_server.tokenizers.scfoundation import scFoundationTokenizer
        try:
            cls.tokenizer = scFoundationTokenizer()
            cls.tokenizer_available = True
        except FileNotFoundError:
            cls.tokenizer_available = False
            cls.tokenizer = None

    def test_gene_vocab_loading(self):
        """Test that gene vocabulary loads correctly with 19,264 genes."""
        if not self.tokenizer_available:
            self.skipTest("Gene vocabulary not available")

        self.assertEqual(len(self.tokenizer.gene_vocab), 19264)
        self.assertEqual(len(self.tokenizer.gene_to_idx), 19264)
        # Check some known genes
        self.assertIn('A1BG', self.tokenizer.gene_vocab)
        self.assertIn('TP53', self.tokenizer.gene_vocab)

    def test_special_tokens(self):
        """Test special token IDs are correctly set."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        self.assertEqual(self.tokenizer.pad_token_id, 19264)
        self.assertEqual(self.tokenizer.mask_token_id, 19265)
        self.assertEqual(self.tokenizer.seq_len, 19264)

    def test_gene_alignment(self):
        """Test gene alignment to vocabulary."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        # Create mock expression data with subset of genes
        n_cells = 5
        test_genes = ['A1BG', 'TP53', 'BRCA1', 'UNKNOWN_GENE']
        expr_matrix = np.random.rand(n_cells,
                                     len(test_genes)).astype(np.float32)

        aligned, missing = self.tokenizer._align_genes(expr_matrix, test_genes)

        # Check output shape
        self.assertEqual(aligned.shape, (n_cells, 19264))

        # Check that known genes have values at correct positions
        a1bg_idx = self.tokenizer.gene_to_idx.get('A1BG')
        if a1bg_idx is not None:
            np.testing.assert_array_almost_equal(aligned[:, a1bg_idx],
                                                 expr_matrix[:, 0])

        # UNKNOWN_GENE should not be in missing (it wasn't in vocab to begin with)
        # Missing genes are vocab genes not in input
        self.assertGreater(len(missing), 0)

    def test_normalization_raw(self):
        """Test log1p(CPM) normalization of raw counts."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        # Raw count data
        expr = np.array([[100, 200, 300], [50, 100, 150]], dtype=np.float32)

        normalized = self.tokenizer._normalize_expression(expr,
                                                          target_sum=10000,
                                                          pre_normalized='F')

        # Check normalization was applied
        self.assertEqual(normalized.shape, expr.shape)

        # Values should be log-transformed
        expected_row0 = np.log1p(expr[0, :] / expr[0, :].sum() * 10000)
        np.testing.assert_array_almost_equal(normalized[0, :],
                                             expected_row0,
                                             decimal=5)

    def test_normalization_prenormalized(self):
        """Test handling of pre-normalized data."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        # Already normalized data
        expr = np.array([[1.5, 2.0, 2.5], [1.0, 1.5, 2.0]], dtype=np.float32)

        normalized = self.tokenizer._normalize_expression(expr,
                                                          target_sum=10000,
                                                          pre_normalized='T')

        # Should be unchanged
        np.testing.assert_array_equal(normalized, expr)

    def test_gather_expressed_genes(self):
        """Test gathering of non-zero (expressed) genes."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        # Create expression with some zeros
        expr = torch.tensor(
            [[0.0, 1.5, 0.0, 2.0, 0.0], [1.0, 0.0, 1.5, 0.0, 2.0]],
            dtype=torch.float32)

        gathered, padding_mask, position_ids = self.tokenizer._gather_expressed_genes(
            expr, pad_token_id=self.tokenizer.pad_token_id)

        # Should gather only non-zero values
        self.assertEqual(gathered.shape[0], 2)  # batch size
        # Max expressed genes per cell is 3
        self.assertLessEqual(gathered.shape[1], 5)

        # Check that non-padded values are correct
        for i in range(2):
            non_pad_mask = ~padding_mask[i]
            non_pad_values = gathered[i][non_pad_mask]
            expected_values = expr[i][expr[i] > 0]
            self.assertEqual(len(non_pad_values), len(expected_values))

    def test_tokenize_cell_vectors_basic(self):
        """Test end-to-end tokenization."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        # Create minimal test data
        n_cells = 3
        test_genes = self.tokenizer.gene_vocab[:100]  # First 100 genes
        expr_matrix = np.random.rand(n_cells,
                                     len(test_genes)).astype(np.float32)
        expr_matrix = expr_matrix * 100  # Scale to count-like values

        # Set some values to zero for sparsity
        expr_matrix[expr_matrix < 50] = 0

        result = self.tokenizer.tokenize_cell_vectors(expr_matrix,
                                                      test_genes,
                                                      target_sum=10000,
                                                      pre_normalized='F',
                                                      return_tensors='pt')

        # Check output keys
        self.assertIn('encoder_data', result)
        self.assertIn('encoder_position_gene_ids', result)
        self.assertIn('encoder_padding_mask', result)

        # Check output types
        self.assertIsInstance(result['encoder_data'], torch.Tensor)
        self.assertIsInstance(result['encoder_position_gene_ids'], torch.Tensor)
        self.assertIsInstance(result['encoder_padding_mask'], torch.Tensor)

        # Check shapes
        self.assertEqual(result['encoder_data'].shape[0], n_cells)
        self.assertEqual(result['encoder_position_gene_ids'].shape[0], n_cells)
        self.assertEqual(result['encoder_padding_mask'].shape[0], n_cells)

    def test_tokenize_with_decoder(self):
        """Test tokenization with decoder data included."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        n_cells = 2
        test_genes = self.tokenizer.gene_vocab[:50]
        expr_matrix = np.random.rand(n_cells, len(test_genes)).astype(
            np.float32) * 100

        result = self.tokenizer.tokenize_cell_vectors(expr_matrix,
                                                      test_genes,
                                                      include_decoder=True,
                                                      return_tensors='pt')

        # Check decoder outputs are present
        self.assertIn('decoder_data', result)
        self.assertIn('decoder_position_gene_ids', result)
        self.assertIn('decoder_padding_mask', result)
        self.assertIn('encoder_labels', result)

    def test_get_config(self):
        """Test configuration retrieval."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        config = self.tokenizer.get_config()

        self.assertEqual(config['pad_token_id'], 19264)
        self.assertEqual(config['mask_token_id'], 19265)
        self.assertEqual(config['num_genes'], 19264)


class TestScFoundationModel(unittest.TestCase):
    """Unit tests for scFoundationModel architecture."""

    def test_model_initialization(self):
        """Test model initializes correctly with default config."""
        from tdc_ml.model_server.models.scfoundation import (scFoundationModel,
                                                             scFoundationConfig)

        model = scFoundationModel()

        self.assertIsNotNone(model.token_emb)
        self.assertIsNotNone(model.pos_emb)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)
        self.assertIsNotNone(model.decoder_embed)
        self.assertIsNotNone(model.norm)
        self.assertIsNotNone(model.to_final)

    def test_model_with_custom_config(self):
        """Test model initialization with custom config."""
        from tdc_ml.model_server.models.scfoundation import (scFoundationModel,
                                                             scFoundationConfig)

        config = scFoundationConfig(encoder_depth=6,
                                    decoder_depth=3,
                                    encoder_hidden_dim=384,
                                    decoder_hidden_dim=256)
        model = scFoundationModel(config)

        # Check encoder dimension
        self.assertEqual(model.config.encoder_hidden_dim, 384)
        self.assertEqual(model.config.encoder_depth, 6)

    def test_auto_discretization_embedding(self):
        """Test AutoDiscretizationEmbedding forward pass."""
        from tdc_ml.model_server.models.scfoundation import AutoDiscretizationEmbedding

        embed = AutoDiscretizationEmbedding(dim=768,
                                            max_seq_len=100,
                                            bin_num=100,
                                            bin_alpha=1.0,
                                            pad_token_id=99,
                                            mask_token_id=100)

        # Test with normal values
        x = torch.randn(2, 50, 1)
        out = embed(x)
        self.assertEqual(out.shape, (2, 50, 768))

        # Test with output weights
        out, weights = embed(x, output_weight=True)
        self.assertEqual(out.shape, (2, 50, 768))
        self.assertEqual(weights.shape, (2, 50, 100))

    def test_fast_attention(self):
        """Test FastAttention forward pass."""
        from tdc_ml.model_server.models.scfoundation import FastAttention

        attn = FastAttention(dim_heads=64, nb_features=128)

        batch_size, heads, seq_len, dim = 2, 8, 100, 64
        q = torch.randn(batch_size, heads, seq_len, dim)
        k = torch.randn(batch_size, heads, seq_len, dim)
        v = torch.randn(batch_size, heads, seq_len, dim)

        out = attn(q, k, v)
        self.assertEqual(out.shape, (batch_size, heads, seq_len, dim))

    def test_self_attention(self):
        """Test SelfAttention module."""
        from tdc_ml.model_server.models.scfoundation import SelfAttention

        attn = SelfAttention(dim=768, heads=12, dim_head=64)

        x = torch.randn(2, 100, 768)
        out = attn(x)
        self.assertEqual(out.shape, (2, 100, 768))

    def test_feed_forward(self):
        """Test FeedForward module."""
        from tdc_ml.model_server.models.scfoundation import FeedForward

        ff = FeedForward(dim=768, mult=4, dropout=0.1)

        x = torch.randn(2, 100, 768)
        out = ff(x)
        self.assertEqual(out.shape, (2, 100, 768))

    def test_performer_module(self):
        """Test PerformerModule forward pass."""
        from tdc_ml.model_server.models.scfoundation import PerformerModule

        performer = PerformerModule(max_seq_len=200,
                                    dim=768,
                                    depth=2,
                                    heads=12,
                                    dim_head=64)

        x = torch.randn(2, 100, 768)
        out = performer(x)
        self.assertEqual(out.shape, (2, 100, 768))

    def test_model_forward_shapes(self):
        """Test model forward pass produces correct output shapes."""
        from tdc_ml.model_server.models.scfoundation import (scFoundationModel,
                                                             scFoundationConfig)

        # Use smaller config for faster testing
        config = scFoundationConfig(encoder_depth=2,
                                    decoder_depth=2,
                                    encoder_hidden_dim=256,
                                    decoder_hidden_dim=128,
                                    encoder_heads=4,
                                    decoder_heads=4,
                                    num_genes=100,
                                    max_seq_len=105)
        model = scFoundationModel(config)

        batch_size = 2
        encoder_seq_len = 20
        decoder_seq_len = 100

        # Create mock inputs
        encoder_data = torch.randn(batch_size, encoder_seq_len)
        encoder_position_ids = torch.arange(encoder_seq_len).unsqueeze(
            0).repeat(batch_size, 1)
        encoder_padding = torch.zeros(batch_size,
                                      encoder_seq_len,
                                      dtype=torch.bool)

        decoder_data = torch.randn(batch_size, decoder_seq_len)
        decoder_position_ids = torch.arange(decoder_seq_len).unsqueeze(
            0).repeat(batch_size, 1)
        decoder_padding = torch.zeros(batch_size,
                                      decoder_seq_len,
                                      dtype=torch.bool)

        encoder_labels = torch.zeros(batch_size,
                                     decoder_seq_len,
                                     dtype=torch.bool)
        encoder_labels[:, :encoder_seq_len] = True

        # Forward pass
        out = model(x=encoder_data,
                    padding_label=encoder_padding,
                    encoder_position_gene_ids=encoder_position_ids,
                    encoder_labels=encoder_labels,
                    decoder_data=decoder_data,
                    decoder_position_gene_ids=decoder_position_ids,
                    decoder_data_padding_labels=decoder_padding)

        self.assertEqual(out.shape, (batch_size, decoder_seq_len))

    def test_get_cell_embedding(self):
        """Test cell embedding extraction."""
        from tdc_ml.model_server.models.scfoundation import (scFoundationModel,
                                                             scFoundationConfig)

        config = scFoundationConfig(encoder_depth=2,
                                    encoder_hidden_dim=256,
                                    encoder_heads=4,
                                    num_genes=100,
                                    max_seq_len=105)
        model = scFoundationModel(config)

        batch_size = 2
        seq_len = 20

        encoder_data = torch.randn(batch_size, seq_len)
        encoder_position_ids = torch.arange(seq_len).unsqueeze(0).repeat(
            batch_size, 1)
        encoder_padding = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Test 'all' pooling
        emb_all = model.get_cell_embedding(encoder_data,
                                           encoder_position_ids,
                                           encoder_padding,
                                           pool_type='all')
        # 4 * encoder_hidden_dim = 4 * 256 = 1024
        self.assertEqual(emb_all.shape, (batch_size, 1024))

        # Test 'max' pooling
        emb_max = model.get_cell_embedding(encoder_data,
                                           encoder_position_ids,
                                           encoder_padding,
                                           pool_type='max')
        self.assertEqual(emb_max.shape, (batch_size, 256))

        # Test 'mean' pooling
        emb_mean = model.get_cell_embedding(encoder_data,
                                            encoder_position_ids,
                                            encoder_padding,
                                            pool_type='mean')
        self.assertEqual(emb_mean.shape, (batch_size, 256))


class TestScFoundationIntegration(unittest.TestCase):
    """Integration tests for scFoundation in PyTDC."""

    @classmethod
    def setUpClass(cls):
        """Check if required files are available."""
        from tdc_ml.model_server.tokenizers.scfoundation import scFoundationTokenizer
        try:
            cls.tokenizer = scFoundationTokenizer()
            cls.tokenizer_available = True
        except FileNotFoundError:
            cls.tokenizer_available = False
            cls.tokenizer = None

    def test_tokenizer_model_integration(self):
        """Test tokenizer output works with model input."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        from tdc_ml.model_server.models.scfoundation import (scFoundationModel,
                                                             scFoundationConfig)

        # Use smaller config for testing
        config = scFoundationConfig(encoder_depth=2,
                                    decoder_depth=2,
                                    encoder_hidden_dim=256,
                                    decoder_hidden_dim=128,
                                    encoder_heads=4,
                                    decoder_heads=4)
        model = scFoundationModel(config)

        # Create test expression data
        n_cells = 3
        test_genes = self.tokenizer.gene_vocab[:500]
        expr_matrix = np.random.rand(n_cells, len(test_genes)).astype(
            np.float32) * 100
        expr_matrix[expr_matrix < 50] = 0  # Add sparsity

        # Tokenize
        tokens = self.tokenizer.tokenize_cell_vectors(expr_matrix,
                                                      test_genes,
                                                      return_tensors='pt')

        # Get cell embeddings
        cell_emb = model.get_cell_embedding(tokens['encoder_data'],
                                            tokens['encoder_position_gene_ids'],
                                            tokens['encoder_padding_mask'],
                                            pool_type='all')

        self.assertEqual(cell_emb.shape[0], n_cells)
        # 4 * 256 = 1024
        self.assertEqual(cell_emb.shape[1], 1024)

    def test_tdc_hf_interface_registration(self):
        """Test that scFoundation is properly registered in tdc_hf_interface."""
        from tdc_ml.model_server.tdc_hf import model_hub

        self.assertIn("scFoundation", model_hub)

    def test_sparse_input_handling(self):
        """Test tokenization handles scipy sparse matrices."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        import scipy.sparse as sp

        n_cells = 5
        test_genes = self.tokenizer.gene_vocab[:100]

        # Create sparse matrix
        dense = np.random.rand(n_cells, len(test_genes)).astype(np.float32)
        dense[dense < 0.5] = 0
        sparse_matrix = sp.csr_matrix(dense)

        # Should handle sparse input
        tokens = self.tokenizer.tokenize_cell_vectors(sparse_matrix,
                                                      test_genes,
                                                      return_tensors='pt')

        self.assertEqual(tokens['encoder_data'].shape[0], n_cells)

    def test_resolution_token_formats(self):
        """Test different resolution token formats."""
        if not self.tokenizer_available:
            self.skipTest("Tokenizer not available")

        n_cells = 2
        test_genes = self.tokenizer.gene_vocab[:50]
        expr_matrix = np.random.rand(n_cells, len(test_genes)).astype(
            np.float32) * 100

        # Test different resolution formats
        for res_format in ['t4', 'f2', 'a1.0']:
            tokens = self.tokenizer.tokenize_cell_vectors(
                expr_matrix,
                test_genes,
                tgt_high_res=res_format,
                return_tensors='pt')
            self.assertEqual(tokens['encoder_data'].shape[0], n_cells)


class TestScFoundationLoader(unittest.TestCase):
    """Unit tests for scFoundationLoader."""

    def test_loader_initialization(self):
        from tdc_ml.model_server.model_loaders.scfoundation_loader import scFoundationLoader

        loader = scFoundationLoader()
        self.assertIsNotNone(loader)
        self.assertTrue(hasattr(loader, 'GDRIVE_FILE_ID'))
        self.assertTrue(hasattr(loader, 'GENE_VOCAB_URL'))

    def test_gdrive_file_id_set(self):
        from tdc_ml.model_server.model_loaders.scfoundation_loader import scFoundationLoader

        loader = scFoundationLoader()
        self.assertIsNotNone(loader.GDRIVE_FILE_ID)
        self.assertIsInstance(loader.GDRIVE_FILE_ID, str)
        self.assertGreater(len(loader.GDRIVE_FILE_ID), 0)

    def test_gene_vocab_url_valid(self):
        """Test that gene vocabulary URL is accessible."""
        from tdc_ml.model_server.model_loaders.scfoundation_loader import scFoundationLoader
        import requests

        loader = scFoundationLoader()

        try:
            response = requests.head(loader.GENE_VOCAB_URL,
                                     allow_redirects=True,
                                     timeout=10)
            self.assertIn(response.status_code, [200, 202, 302, 303])
        except requests.exceptions.RequestException:
            self.skipTest("Network unavailable")


if __name__ == '__main__':
    unittest.main()
