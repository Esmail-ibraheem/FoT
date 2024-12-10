"""Test dynamic parallelism with Megatron models."""

import os
import sys
import torch
import unittest
import random
import numpy as np
from functools import partial

# Add Megatron path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.path.pardir)))

from megatron import get_args
from megatron import get_tokenizer
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel, BertModel
from megatron.mpu import destroy_model_parallel
from megatron.mpu import initialize_model_parallel
from megatron.mpu import reinitialize_model_parallel
from megatron.training import get_model
from megatron.checkpointing import save_checkpoint
from megatron.checkpointing import load_checkpoint

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class TestDynamicTraining(unittest.TestCase):
    """Test suite for dynamic parallel training with Megatron models."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Megatron and set up basic configurations."""
        # Basic args for testing
        sys.argv = ['script.py',
                   '--num-layers', '4',
                   '--hidden-size', '256',
                   '--num-attention-heads', '8',
                   '--micro-batch-size', '2',
                   '--global-batch-size', '8',
                   '--seq-length', '32',
                   '--max-position-embeddings', '32',
                   '--train-iters', '20',
                   '--lr-decay-iters', '16',
                   '--data-impl', 'mock',
                   '--distributed-backend', 'nccl',
                   '--lr', '0.0001',
                   '--lr-decay-style', 'linear',
                   '--min-lr', '1.0e-5',
                   '--weight-decay', '1e-2',
                   '--clip-grad', '1.0',
                   '--lr-warmup-fraction', '.01',
                   '--vocab-file', 'tests/dummy_data/gpt2-vocab.json',
                   '--merge-file', 'tests/dummy_data/gpt2-merges.txt',
                   '--checkpoint-activations']
                   
        initialize_megatron()
        cls.args = get_args()
        cls.tokenizer = get_tokenizer()
        set_random_seed(123)

    def setUp(self):
        """Set up each test."""
        set_random_seed(123)
        
    def tearDown(self):
        """Clean up after each test."""
        destroy_model_parallel()
        torch.distributed.barrier()

    def test_gpt_dynamic_parallel_transition(self):
        """Test GPT model with dynamic parallel strategy changes."""
        # Initial parallel setup
        initialize_model_parallel(tensor_parallel_size=2)
        
        # Get initial model
        model = get_model(lambda: GPTModel(
            num_layers=self.args.num_layers,
            vocab_size=self.args.padded_vocab_size,
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            embedding_dropout_prob=self.args.hidden_dropout,
            attention_dropout_prob=self.args.attention_dropout,
            output_dropout_prob=self.args.hidden_dropout,
            max_sequence_length=self.args.max_position_embeddings,
            checkpoint_activations=self.args.checkpoint_activations,
            checkpoint_num_layers=self.args.checkpoint_num_layers,
            parallel_output=True))

        # Create dummy batch
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, self.args.padded_vocab_size, 
                                (batch_size, seq_length), 
                                device='cuda')
        attention_mask = torch.ones((batch_size, seq_length), 
                                  dtype=torch.float, 
                                  device='cuda')
        
        # Initial forward pass
        initial_output = model(input_ids, attention_mask, labels=None)
        initial_logits = initial_output[0]
        
        # Save initial state
        save_checkpoint(0, model, None, None)
        
        # Change parallel strategy
        reinitialize_model_parallel(tensor_parallel_size=4)
        model.update_parallel_state()
        
        # Forward pass after strategy change
        new_output = model(input_ids, attention_mask, labels=None)
        new_logits = new_output[0]
        
        # Verify outputs have same shape
        self.assertEqual(initial_logits.shape, new_logits.shape)
        
        # Load checkpoint and verify
        iteration = load_checkpoint(model, None, None)
        self.assertEqual(iteration, 0)
        
        # Forward pass after loading
        loaded_output = model(input_ids, attention_mask, labels=None)
        loaded_logits = loaded_output[0]
        
        # Verify loaded model produces same output shape
        self.assertEqual(initial_logits.shape, loaded_logits.shape)

    def test_bert_dynamic_parallel_transition(self):
        """Test BERT model with dynamic parallel strategy changes."""
        # Initial parallel setup
        initialize_model_parallel(tensor_parallel_size=2)
        
        # Get initial model
        model = get_model(lambda: BertModel(
            num_layers=self.args.num_layers,
            vocab_size=self.args.padded_vocab_size,
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            max_sequence_length=self.args.max_position_embeddings,
            embedding_dropout_prob=self.args.hidden_dropout,
            attention_dropout_prob=self.args.attention_dropout,
            output_dropout_prob=self.args.hidden_dropout,
            checkpoint_activations=self.args.checkpoint_activations,
            checkpoint_num_layers=self.args.checkpoint_num_layers))

        # Create dummy batch
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, self.args.padded_vocab_size, 
                                (batch_size, seq_length), 
                                device='cuda')
        attention_mask = torch.ones((batch_size, seq_length), 
                                  dtype=torch.float, 
                                  device='cuda')
        token_type_ids = torch.zeros((batch_size, seq_length), 
                                   dtype=torch.long, 
                                   device='cuda')
        
        # Initial forward pass
        initial_output = model(input_ids, attention_mask, token_type_ids)
        initial_sequence_output = initial_output[0]
        
        # Save initial state
        save_checkpoint(0, model, None, None)
        
        # Change parallel strategy
        reinitialize_model_parallel(tensor_parallel_size=4)
        model.update_parallel_state()
        
        # Forward pass after strategy change
        new_output = model(input_ids, attention_mask, token_type_ids)
        new_sequence_output = new_output[0]
        
        # Verify outputs have same shape
        self.assertEqual(initial_sequence_output.shape, new_sequence_output.shape)
        
        # Load checkpoint and verify
        iteration = load_checkpoint(model, None, None)
        self.assertEqual(iteration, 0)
        
        # Forward pass after loading
        loaded_output = model(input_ids, attention_mask, token_type_ids)
        loaded_sequence_output = loaded_output[0]
        
        # Verify loaded model produces same output shape
        self.assertEqual(initial_sequence_output.shape, loaded_sequence_output.shape)

    def test_dynamic_training_loop(self):
        """Test full training loop with dynamic parallel changes."""
        # Initial parallel setup
        initialize_model_parallel(tensor_parallel_size=2)
        
        # Get model
        model = get_model(lambda: GPTModel(
            num_layers=self.args.num_layers,
            vocab_size=self.args.padded_vocab_size,
            hidden_size=self.args.hidden_size,
            num_attention_heads=self.args.num_attention_heads,
            embedding_dropout_prob=self.args.hidden_dropout,
            attention_dropout_prob=self.args.attention_dropout,
            output_dropout_prob=self.args.hidden_dropout,
            max_sequence_length=self.args.max_position_embeddings,
            checkpoint_activations=self.args.checkpoint_activations,
            checkpoint_num_layers=self.args.checkpoint_num_layers,
            parallel_output=True))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Training loop with dynamic changes
        for step in range(10):
            # Create dummy batch
            batch_size = 2
            seq_length = 32
            input_ids = torch.randint(0, self.args.padded_vocab_size, 
                                    (batch_size, seq_length), 
                                    device='cuda')
            attention_mask = torch.ones((batch_size, seq_length), 
                                      dtype=torch.float, 
                                      device='cuda')
            labels = torch.randint(0, self.args.padded_vocab_size, 
                                 (batch_size, seq_length), 
                                 device='cuda')
            
            # Forward pass
            output = model(input_ids, attention_mask, labels=labels)
            loss = output[0]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Change parallel strategy every 5 steps
            if step == 5:
                # Save checkpoint
                save_checkpoint(step, model, optimizer, None)
                
                # Change strategy
                reinitialize_model_parallel(tensor_parallel_size=4)
                model.update_parallel_state()
                
                # Load checkpoint
                iteration = load_checkpoint(model, optimizer, None)
                self.assertEqual(iteration, step)

if __name__ == '__main__':
    unittest.main()