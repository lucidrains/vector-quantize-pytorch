# Example demonstrating ResidualFSQ with QINCO (Quantized Implicit Neural Codebooks)
# Based on "Residual Quantization with Implicit Neural Codebooks" https://arxiv.org/abs/2401.14732

import torch
import torch.nn.functional as F
from vector_quantize_pytorch.residual_fsq import ResidualFSQ

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a synthetic latent to be quantized
batch_size = 4
latent_dim = 512 # Following paper recommendations from section 4.1 and A.4.1
seq_length = 32
input_latent = torch.randn(batch_size, seq_length, latent_dim)

# Define FSQ levels for each dimension - typical values from the paper
levels = [8, 5, 5, 5]  # As used in the FSQ paper section 4.1 and A.4.1

print("=== Testing ResidualFSQ with QINCO ===\n")

# ResidualFSQ with QINCO
qinco_residual_fsq = ResidualFSQ(
    levels=levels,
    num_quantizers=4,
    dim=latent_dim,
    quantize_dropout=True,
    quantize_dropout_cutoff_index=1,
    preserve_symmetry=True,
    implicit_neural_codebook=True,  # Enable QINCO
    mlp_kwargs=dict(
        dim_hidden=64,
        depth=4
    )
)

# Run model
with torch.no_grad():
    qinco_output, qinco_indices = qinco_residual_fsq(input_latent)

# Calculate MSE
qinco_mse = F.mse_loss(qinco_output, input_latent).item()

print(f"QINCO ResidualFSQ MSE: {qinco_mse:.6f}")
print(f"Input latent shape: {input_latent.shape}")
print(f"QINCO output shape: {qinco_output.shape}")
print(f"QINCO indices shape: {qinco_indices.shape}")

# Test reconstruction from indices
with torch.no_grad():
    # Reconstruct from indices
    qinco_reconstructed = qinco_residual_fsq.get_output_from_indices(qinco_indices)
    
    # Calculate reconstruction MSE
    qinco_recon_mse = F.mse_loss(qinco_reconstructed, input_latent).item()
    
    print(f"QINCO Reconstruction MSE: {qinco_recon_mse:.6f}")
    
    # Check if reconstructions match the original outputs
    qinco_output_match = F.mse_loss(qinco_reconstructed, qinco_output).item()
    
    print(f"QINCO output vs reconstruction MSE: {qinco_output_match:.6f}")
