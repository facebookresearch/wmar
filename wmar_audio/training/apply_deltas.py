# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders, LMGen

def apply_delta_to_model(original_model, delta_path, component_type="encoder"):
    """
    Apply delta weights to an original model to reconstruct the finetuned model.
    
    Args:
        original_model: The original model (e.g., mimi_ori, or wrapper model)
        delta_path: Path to the delta .pth file
        component_type: "encoder" or "decoder" for the component to update
    
    Returns:
        The model with updated weights
    """
    import copy
    
    print(f"Loading delta from: {delta_path}")
    delta_state_dict = torch.load(delta_path, map_location="cpu", weights_only=False)
    
    # Create a copy of the original model to avoid modifying it
    updated_model = copy.deepcopy(original_model)
    
    # Determine which component to update based on the model type
    if hasattr(updated_model, component_type): 
        component = getattr(updated_model, component_type)

    # Apply deltas to the component
    original_state_dict = component.state_dict()
    updated_state_dict = {}
    
    for key in delta_state_dict:
        if key in original_state_dict:
            updated_state_dict[key] = original_state_dict[key] + delta_state_dict[key]
        else:
            raise ValueError(f"Key {key} not found in original {component_type} state dict")
    
    # Load the updated weights
    component.load_state_dict(updated_state_dict)
    
    print(f"Successfully applied delta to {component_type}")
    return updated_model


def reconstruct_moshi_from_delta(encoder_delta_path=None, decoder_delta_path=None):
    """
    Reconstruct a finetuned Moshi model from delta files.
    
    Args:
        encoder_delta_path: Path to encoder delta file (optional)
        decoder_delta_path: Path to decoder delta file (optional)
    
    Returns:
        Reconstructed Moshi model with applied deltas
    """
    if loaders is None:
        raise ImportError("moshi.models not available")
    
    # Load original MIMI model
    mimi_weight_ori = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)
    print(f"Loading original MIMI from: {mimi_weight_ori}")
    mimi_reconstructed = loaders.get_mimi(mimi_weight_ori, "cpu")
    
    # Apply encoder delta if provided
    if encoder_delta_path and os.path.exists(encoder_delta_path):
        mimi_reconstructed = apply_delta_to_model(mimi_reconstructed, encoder_delta_path, "encoder")
    
    # Apply decoder delta if provided
    if decoder_delta_path and os.path.exists(decoder_delta_path):
        mimi_reconstructed = apply_delta_to_model(mimi_reconstructed, decoder_delta_path, "decoder")
    
    return mimi_reconstructed


def main():
    """
    Main function to handle command-line arguments and save the reconstructed model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply delta weights to reconstruct finetuned MIMI model")
    parser.add_argument("--encoder_delta", type=str, default=None,
                        help="Path to encoder delta file (.pth)")
    parser.add_argument("--decoder_delta", type=str, default=None,
                        help="Path to decoder delta file (.pth)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetunes",
                        help="Directory to save the reconstructed model (default: checkpoints/finetunes)")
    parser.add_argument("--output_name", type=str, default="mimi_reconstructed.pth",
                        help="Name of the output file (default: mimi_reconstructed.pth)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.encoder_delta and not args.decoder_delta:
        raise ValueError("At least one of --encoder_delta or --decoder_delta must be provided")
    
    if args.encoder_delta and not os.path.exists(args.encoder_delta):
        raise FileNotFoundError(f"Encoder delta file not found: {args.encoder_delta}")
    
    if args.decoder_delta and not os.path.exists(args.decoder_delta):
        raise FileNotFoundError(f"Decoder delta file not found: {args.decoder_delta}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Reconstruct the model
    print("Starting model reconstruction...")
    mimi_reconstructed = reconstruct_moshi_from_delta(
        encoder_delta_path=args.encoder_delta,
        decoder_delta_path=args.decoder_delta
    )
    
    # Save the reconstructed model state_dict
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"Saving reconstructed model to: {output_path}")
    
    torch.save({"model": mimi_reconstructed.state_dict()}, output_path)
    
    print("Model reconstruction and saving completed successfully!")
    print(f"Reconstructed model saved at: {output_path}")
    
    # Print summary of what was applied
    if args.encoder_delta:
        print(f"Applied encoder delta from: {args.encoder_delta}")
    if args.decoder_delta:
        print(f"Applied decoder delta from: {args.decoder_delta}")


if __name__ == "__main__":
    main()

