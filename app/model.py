import torch
from torch import nn
from torchvision import models

def create_foodvision_model(num_classes: int = 101):
    """Creates a pretrained Vision Transformer (ViT-Base/32) model with a custom classifier head.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 101 (for Food101 dataset).

    Returns:
        model (torch.nn.Module): Vision Transformer model.
    """
    # Load a pretrained Vision Transformer (ViT-Base/32) model
    model = models.vit_b_32(weights='IMAGENET1K_V1')

    # Freeze all layers in base model first
    for param in model.parameters():
      param.requires_grad = False

    # Replace the classifier head with a new one for num_classes
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(num_ftrs),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )

    # Initialize the weights of the new classifier head
    try:
        torch.nn.init.xavier_uniform_(model.heads.head[2].weight)
        model.heads.head[2].bias.data.fill_(0.01)
    except IndexError:
         print("Warning: Could not initialize weights for the new head. Ensure the head structure is as expected.")


    # Unfreeze the last few transformer blocks (e.g., last 2-4) as done in the notebook
    # The ViT encoder has 12 blocks, indexing from 0 to 11
    num_unfreeze_blocks = 4 # This should match the value used in the notebook if fine-tuned
    for i in range(1, num_unfreeze_blocks + 1):
        for param in model.encoder.layers[-i].parameters():
            param.requires_grad = True

    # Unfreeze the LayerNorm before the head as done in the notebook
    for param in model.encoder.ln.parameters():
        param.requires_grad = True

    # Unfreeze the new classifier head
    for param in model.heads.parameters():
      param.requires_grad = True


    return model

# Example usage (optional, can be removed for just the model definition)
# if __name__ == '__main__':
#     # Create an instance of the model
#     food_model = create_foodvision_model(num_classes=101)
#     print("Food Vision model created:")
#     print(food_model)
#
#     # Example of moving to device (assuming 'device' is defined elsewhere)
#     # food_model.to(device)
#     # print(f"Model moved to {device}")
