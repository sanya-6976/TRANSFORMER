from tasks.model.transformer_stack import TransformerStack


# Model configuration
embedding_dim = 128
num_heads = 4
num_layers = 6

model = TransformerStack(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    expansion_factor=4,
    dropout=0.0
)

# Count parameters in a module
def count_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

# Analyze each Transformer block
for index, block in enumerate(model.blocks):

    print(f"\nTransformer Block {index + 1}")

    attention_params = count_parameters(block.attention)

    ffn_params = count_parameters(block.feed_forward)

    ln1_params = count_parameters(block.layer_norm1)

    ln2_params = count_parameters(block.layer_norm2)

    total = count_parameters(block)

    print("Attention parameters:", attention_params)
    print("FFN parameters:", ffn_params)
    print("LayerNorm 1 parameters:", ln1_params)
    print("LayerNorm 2 parameters:", ln2_params)
    print("Total block parameters:", total)


# Final LayerNorm
final_ln_params = count_parameters(model.final_layer_norm)

print("\nFinal LayerNorm parameters:", final_ln_params)


# Entire stack
total_params = count_parameters(model)

print("Total Transformer Stack parameters:", total_params)