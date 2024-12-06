import torch
import matplotlib.pyplot as plt
from fbm_dropout_function_for_cnn import DropoutFBM
from torchvision import transforms

def visualize_fbm_dropout(hurst, n_agents, n_samples, max_iters, t_scale, grid_size, is_conv, num_steps=5):
    """
    Parameters:
    - hurst (float): Hurst parameter for FBM. (+) Smoothness of fiber paths
    - n_agents (int): Number of FBM fibers.
    - n_samples (int): Number of samples per iteration. (+) Path length per iteration
    - max_iters (int): Maximum iterations for FBM. (+) Total potential length
    - t_scale (float): Scale parameter for FBM.
    - grid_size (tuple): Size of the grid.
    - is_conv (bool): Is convolutional layer.
    - num_steps (int): Number of simulations.
    """
    # Initialize the DropoutFBM layer
    dropout_fbm = DropoutFBM(
        hurst=hurst,
        n_agents=n_agents,
        n_samples=n_samples,
        max_iters=max_iters,
        t_scale=t_scale,
        grid_size=grid_size,
        is_conv=is_conv
    )

    # Simulate an input tensor
    if is_conv:
        input_tensor = torch.randn((1, 3, grid_size[0], grid_size[1]))  # Simulating 3 feature maps
    else:
        input_tensor = torch.randn((grid_size[0] * grid_size[1],))

    # Visualize over iterations
    print(f"Number of fibers: {n_agents}")
    print(f"Samples per iteration: {n_samples}")
    print(f"Maximum iterations: {max_iters}")
    print(f"Hurst: {hurst:.2f}")
    for step in range(1, num_steps + 1):
        with torch.no_grad():
            output_tensor = dropout_fbm(input_tensor)

        print(f"Simluation {step}: Dropout Rate: {dropout_fbm.now_dropout_rate:.2f}")

        title = f"FBM Dropout at Step {step}"
        dropout_fbm.show_grid(title=title, dpi=150, show_history=True)

if __name__ == "__main__":
    visualize_fbm_dropout(
        hurst=0.9,
        n_agents=35,
        n_samples=1000,
        max_iters=40,
        t_scale=1.0,
        grid_size=(16, 16),  
        is_conv=True,
        num_steps=3
    )