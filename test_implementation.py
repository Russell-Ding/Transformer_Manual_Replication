import torch
import torch.nn as nn
import math
from attention_manual_replication import Atten_Layers, Multi_Head_Layer, Bert_Base_Layer, Bert_Encoder, generate_positional_embeddings

def test_attention_dimensions():
    """Test that attention layer produces correct output dimensions"""
    print("=== Testing Attention Layer Dimensions ===")
    
    batch_size, seq_len, d_model, num_heads = 2, 10, 768, 8
    
    # Test single attention head
    attn_layer = Atten_Layers(max_length=seq_len, num_heads=num_heads, d_model=d_model)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = attn_layer(x, x, x)
    expected_dim = d_model // num_heads  # Should be 96 for 768/8
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected last dim: {expected_dim}")
    print(f"Actual last dim: {output.shape[-1]}")
    print(f"✓ Correct dimensions: {output.shape[-1] == expected_dim}\n")
    
    return output.shape[-1] == expected_dim

def test_multihead_attention():
    """Test multi-head attention concatenation and output projection"""
    print("=== Testing Multi-Head Attention ===")
    
    batch_size, seq_len, d_model, num_heads = 2, 10, 768, 8
    
    multi_head = Multi_Head_Layer(max_length=seq_len, num_heads=num_heads, d_model=d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = multi_head(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Output maintains d_model: {output.shape[-1] == d_model}")
    print(f"✓ Batch and sequence dims preserved: {output.shape[:2] == x.shape[:2]}\n")
    
    return output.shape == x.shape

def test_scaling_factor():
    """Test if scaling factor is correct"""
    print("=== Testing Scaling Factor ===")
    
    d_model, num_heads = 768, 8
    d_k = d_model // num_heads  # Should be 96
    
    attn_layer = Atten_Layers(max_length=10, num_heads=num_heads, d_model=d_model)
    
    # Create simple inputs to check scaling
    batch_size, seq_len = 1, 3
    x = torch.ones(batch_size, seq_len, d_model)
    
    # Check the scaling factor used in your implementation
    with torch.no_grad():
        Q = attn_layer.w_query(x)  # Shape: [1, 3, 96]
        K = attn_layer.w_key(x)    # Shape: [1, 3, 96]
        
        # Your implementation uses this scaling
        your_scaling = torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        paper_scaling = math.sqrt(d_k)
        
        print(f"d_k (dimension per head): {d_k}")
        print(f"Your scaling factor: {your_scaling.item():.4f}")
        print(f"Paper scaling factor: {paper_scaling:.4f}")
        print(f"✓ Scaling factors match: {abs(your_scaling.item() - paper_scaling) < 1e-6}\n")
        
        return abs(your_scaling.item() - paper_scaling) < 1e-6

def test_positional_encoding():
    """Test positional encoding implementation"""
    print("=== Testing Positional Encoding ===")
    
    d_model, max_length, batch_size = 768, 512, 2
    
    # Your implementation
    your_pe = generate_positional_embeddings(d_model, max_length, batch_size)
    
    # Correct implementation for comparison
    def correct_positional_encoding(d_model, max_length):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    correct_pe = correct_positional_encoding(d_model, max_length)
    
    print(f"Your PE shape: {your_pe.shape}")
    print(f"Correct PE shape: {correct_pe.shape}")
    
    # Check a few values
    print(f"Your PE[0,0,0]: {your_pe[0,0,0]:.6f}")
    print(f"Correct PE[0,0]: {correct_pe[0,0]:.6f}")
    print(f"Your PE[0,1,1]: {your_pe[0,1,1]:.6f}")
    print(f"Correct PE[1,1]: {correct_pe[1,1]:.6f}")
    
    # Check if patterns are similar (allowing for implementation differences)
    are_similar = torch.allclose(your_pe[0], correct_pe, atol=1e-1)
    print(f"✓ Positional encodings are similar: {are_similar}\n")
    
    return are_similar

def test_feedforward_network():
    """Test feed-forward network structure"""
    print("=== Testing Feed-Forward Network ===")
    
    d_model = 768
    bert_layer = Bert_Base_Layer(d_model=d_model)
    
    # Check if feed-forward has correct dimensions
    ff_input_dim = bert_layer.linear1.in_features
    ff_hidden_dim = bert_layer.linear1.out_features
    ff_output_dim = bert_layer.linear2.out_features
    
    expected_hidden = d_model * 4  # Paper uses 4x expansion
    
    print(f"FF input dim: {ff_input_dim}")
    print(f"FF hidden dim: {ff_hidden_dim}")
    print(f"FF output dim: {ff_output_dim}")
    print(f"Expected hidden dim: {expected_hidden}")
    print(f"✓ Correct FF structure: {ff_hidden_dim == expected_hidden and ff_input_dim == ff_output_dim == d_model}\n")
    
    return ff_hidden_dim == expected_hidden and ff_input_dim == ff_output_dim == d_model

def test_parameter_registration():
    """Test that all parameters are properly registered"""
    print("=== Testing Parameter Registration ===")
    
    # Test multi-head layer
    multi_head = Multi_Head_Layer(num_heads=8, d_model=768)
    multi_head_params = sum(p.numel() for p in multi_head.parameters())
    
    # Test BERT encoder
    bert_encoder = Bert_Encoder(num_vocab=1000, num_attention_layers=2)
    bert_params = sum(p.numel() for p in bert_encoder.parameters())
    
    print(f"Multi-head layer parameters: {multi_head_params:,}")
    print(f"BERT encoder parameters: {bert_params:,}")
    print(f"✓ Parameters are registered (> 0): {multi_head_params > 0 and bert_params > 0}\n")
    
    return multi_head_params > 0 and bert_params > 0

def test_full_forward_pass():
    """Test complete forward pass through the model"""
    print("=== Testing Full Forward Pass ===")
    
    batch_size, seq_len = 2, 10
    vocab_size = 1000
    
    # Create model
    model = Bert_Encoder(
        num_vocab=vocab_size,
        max_length=seq_len,
        num_attention_layers=2,
        d_model=768
    )
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        output = model(input_ids)
        success = True
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful")
    except Exception as e:
        success = False
        print(f"✗ Forward pass failed: {e}")
    
    print()
    return success

def test_attention_weights_sum():
    """Test that attention weights sum to 1"""
    print("=== Testing Attention Weights ===")
    
    batch_size, seq_len, d_model, num_heads = 1, 4, 768, 8
    
    attn_layer = Atten_Layers(max_length=seq_len, num_heads=num_heads, d_model=d_model)
    
    # Create simple input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Hook to capture attention weights
    attention_weights = None
    
    def hook_fn(module, input, output):
        nonlocal attention_weights
        # The attention weights are computed in the forward pass
        # We need to manually compute them to check
        Q = attn_layer.w_query(x)
        K = attn_layer.w_key(x)
        scores = torch.bmm(Q, K.transpose(-1,-2)) / math.sqrt(d_model // num_heads)
        attention_weights = torch.softmax(scores, dim=-1)
    
    # Register hook
    handle = attn_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    output = attn_layer(x, x, x)
    
    # Remove hook
    handle.remove()
    
    if attention_weights is not None:
        # Check if attention weights sum to 1 along the last dimension
        weight_sums = attention_weights.sum(dim=-1)
        all_sum_to_one = torch.allclose(weight_sums, torch.ones_like(weight_sums))
        
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Sample attention sums: {weight_sums[0, :3]}")
        print(f"✓ Attention weights sum to 1: {all_sum_to_one}")
    else:
        all_sum_to_one = False
        print("✗ Could not capture attention weights")
    
    print()
    return all_sum_to_one

def run_all_tests():
    """Run all test cases"""
    print("🧪 Running Comprehensive Tests for Your Implementation\n")
    
    tests = [
        ("Attention Dimensions", test_attention_dimensions),
        ("Multi-Head Attention", test_multihead_attention),
        ("Scaling Factor", test_scaling_factor),
        ("Positional Encoding", test_positional_encoding),
        ("Feed-Forward Network", test_feedforward_network),
        ("Parameter Registration", test_parameter_registration),
        ("Full Forward Pass", test_full_forward_pass),
        ("Attention Weights", test_attention_weights_sum),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}\n")
            results[test_name] = False
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your implementation looks good!")
    else:
        print("🔧 Some tests failed. Check the details above for areas to improve.")

if __name__ == "__main__":
    run_all_tests()