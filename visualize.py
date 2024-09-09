from torchviz import make_dot
import torch

def visualize_model(model, example_input):
    model.eval()
    with torch.no_grad():
        # Forward pass to get the output
        combined_output, expert_outputs, gating_weights = model(example_input)
        
        # Create a graph of the model
        dot = make_dot(combined_output, params=dict(model.named_parameters()))
        dot.render('model_architecture', format='png')  # Save the visualization as a PNG file
        dot.view()  # Open the PNG file

# Create a dummy input to pass through the model
example_input = torch.tensor([encode_text('example')]).to(device)
visualize_model(model, example_input)
