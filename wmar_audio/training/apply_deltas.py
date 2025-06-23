# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders

def apply_delta_to_model(original_model, delta_state_dict):
    """
    Apply delta weights to an original model to reconstruct the finetuned model.
    Args:
        original_model: The original model (e.g., mimi_ori)
        delta_state_dict: Flat dictionary containing deltas with component prefixes
    Returns:
        The model with updated weights
    """
    import copy
    
    print("Applying deltas...")
    
    # Create a copy of the original model to avoid modifying it
    updated_model = copy.deepcopy(original_model)
    
    # Group deltas by component
    components = ["encoder", "decoder", "encoder_transformer", "decoder_transformer"]
    
    for component_name in components:
        if hasattr(updated_model, component_name):
            component = getattr(updated_model, component_name)
            component_state_dict = component.state_dict()
            updated_component_state_dict = {}
            
            # Find deltas for this component
            component_deltas = {k: v for k, v in delta_state_dict.items() if k.startswith(f"{component_name}.")}
            
            if component_deltas:
                for key in component_state_dict:
                    delta_key = f"{component_name}.{key}"
                    if delta_key in delta_state_dict:
                        updated_component_state_dict[key] = component_state_dict[key] + delta_state_dict[delta_key]
                    else:
                        updated_component_state_dict[key] = component_state_dict[key]
                
                # Load updated weights
                component.load_state_dict(updated_component_state_dict)
                
                applied_deltas = len(component_deltas)
                print(f"Applied {applied_deltas} deltas to {component_name}")
    
    return updated_model


def reconstruct_moshi_from_delta(delta_path):
    """
    Reconstruct a finetuned Moshi model from delta checkpoint file.
    
    Args:
        delta_path: Path to delta checkpoint file
    
    Returns:
        Reconstructed Moshi model with applied deltas
    """
    # Load original MIMI model
    mimi_weight_ori = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)
    print(f"Loading original MIMI from: {mimi_weight_ori}")
    mimi_reconstructed = loaders.get_mimi(mimi_weight_ori, "cpu")
    
    # Load delta checkpoint
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"Delta checkpoint not found: {delta_path}")
    
    print(f"Loading delta checkpoint from: {delta_path}")
    delta_state_dict = torch.load(delta_path, map_location="cpu", weights_only=False)
    
    # Apply deltas
    mimi_reconstructed = apply_delta_to_model(mimi_reconstructed, delta_state_dict)
    
    return mimi_reconstructed


def main():
    """
    Main function to handle command-line arguments and save the reconstructed model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply delta weights to reconstruct finetuned MIMI model")
    parser.add_argument("--delta_path", type=str, required=True,
                        help="Path to _path checkpoint file (.pth)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetunes",
                        help="Directory to save the reconstructed model (default: checkpoints/finetunes)")
    parser.add_argument("--output_name", type=str, default="mimi_reconstructed.pth",
                        help="Name of the output file (default: mimi_reconstructed.pth)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.delta_path):
        raise FileNotFoundError(f"Delta checkpoint file not found: {args.delta_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Reconstruct the model
    print("Starting model reconstruction...")
    mimi_reconstructed = reconstruct_moshi_from_delta(args.delta_path)
    
    # Save the reconstructed model state_dict
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"Saving reconstructed model to: {output_path}")
    
    torch.save({"model": mimi_reconstructed.state_dict()}, output_path)
    
    print("Model reconstruction and saving completed successfully!")
    print(f"Reconstructed model saved at: {output_path}")
    print(f"Applied deltas from: {args.delta_path}")


if __name__ == "__main__":
    main()