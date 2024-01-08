import torch
from latent_quantization import LatentQuantize
from finite_scalar_quantization import FSQ

def test_single_codebook_equal_levels():
    levels = 5
    dim = 8
    num_codebooks = 1
    model = LatentQuantize(levels, dim, num_codebooks=num_codebooks)
    input_tensor = torch.randn(2, 3, dim)
    output_tensor, indices, loss = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert indices.shape == (2, 3)
    assert loss.item() >= 0

def test_multiple_codebooks_different_levels():
    levels = [4, 8, 16]
    dim = 9
    num_codebooks = 3
    model = LatentQuantize(levels, dim, num_codebooks=num_codebooks)
    input_tensor = torch.randn(2, 3, dim)
    output_tensor, indices, loss = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert indices.shape == (2, 3, num_codebooks)
    assert loss.item() >= 0

def test_zero_commitment_and_quantization_loss_weight():
    levels = 8
    dim = 4
    num_codebooks = 1
    model = LatentQuantize(levels, dim, num_codebooks=num_codebooks, commitment_loss_weight=0, quantization_loss_weight=0)
    input_tensor = torch.randn(2, 1, dim)
    output_tensor, indices, loss = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert indices.shape == (2, 1)
    assert loss.item() == 0

def test_images():
    levels = 5
    dim = 4
    num_codebooks = 1
    model = LatentQuantize(levels, dim, num_codebooks=num_codebooks, commitment_loss_weight=0)
    input_tensor = torch.randn(2, dim, 32, 32)
    output_tensor, indices, loss = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert indices.shape == (2, 32, 32)

def test_videos():
    levels = 5
    dim = 6
    num_codebooks = 2
    model = LatentQuantize(levels, dim, num_codebooks=num_codebooks, commitment_loss_weight=0)
    input_tensor = torch.randn(2, 6, dim, 32, 32)
    output_tensor, indices, loss = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert indices.shape == (2, 6, 32, 32, num_codebooks)

def test_latent_quantize_readme1():
    quantizer = LatentQuantize(

        levels = [8,5,8],      # number of levels for each codebook dimension
        dim = 16,                   # dimension of input tensor
        commitment_loss_weight=0.1,  # how much weight to place on commitment loss
        quantization_loss_weight=0.1, # how much weight to place on quantization loss
    )

    image_feats = torch.randn(1, 16, 32, 32)

    quantized, indices, loss = quantizer(image_feats)

    # (1, 16, 32, 32), (1, 32, 32), (1,)

    assert image_feats.shape == quantized.shape
    assert torch.allclose(quantized, quantizer.indices_to_codes(indices), atol=1e-5)

def test_latent_quantize_readme2():
    quantizer = LatentQuantize(

        levels = [5,5,7],      # 
        dim = 16,                   # 
    )

    seq = torch.randn(1, 32, 16)

    quantized, indices, loss = quantizer(seq)

    # (1, 32, 16), (1, 32), (1,)

    assert seq.shape == quantized.shape
    assert torch.allclose(quantized, quantizer.indices_to_codes(indices), atol=1e-3)

def test_latent_quantize_readme3():
    quantizer = LatentQuantize(

        levels = [6,5,7],  
        dim = 16,               
    )

    seq = torch.randn(1, 16, 10, 32, 32)

    quantized, indices, loss = quantizer(seq)

    # (1, 16, 10, 32, 32), (1, 10, 32, 32), (1,)

    assert seq.shape == quantized.shape
    assert torch.allclose(quantized, quantizer.indices_to_codes(indices), atol=1e-5)

#test_single_codebook_equal_levels()
#test_multiple_codebooks_different_levels()
#test_zero_commitment_and_quantization_loss_weight()
#test_images()
#test_videos()
test_latent_quantize_readme1()
test_latent_quantize_readme2()
test_latent_quantize_readme3()

print("All test cases passed!")
