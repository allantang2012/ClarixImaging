import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import scalar

def read_tensorboard_data(log_dir):
    """Read tensorboard event files and extract loss values."""
    losses = {}

    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, 'default', 'events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")

    # Use the most recent file
    latest_file = max(event_files, key=os.path.getctime)
    print(f"Reading from {latest_file}")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(latest_file)
    ea.Reload()

    # Get all scalar tags (loss components)
    tags = ea.Tags()['scalars']

    # Extract values for each tag
    for tag in tags:
        events = ea.Scalars(tag)
        losses[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }

    return losses

def plot_losses(losses, output_file='loss_plot.png'):
    """Plot loss curves."""
    plt.figure(figsize=(12, 6))

    for tag, data in losses.items():
        # Plot with moving average for smoothing
        values = np.array(data['values'])
        steps = np.array(data['steps'])

        # Calculate moving average
        window_size = min(50, len(values))
        if window_size > 0:
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[window_size-1:]
            plt.plot(smoothed_steps, smoothed, label=tag, alpha=0.8)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses (50-iteration moving average)')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

    # Print final loss values
    print("\nFinal loss values:")
    for tag, data in losses.items():
        if data['values']:
            print(f"{tag}: {data['values'][-1]:.4f}")

if __name__ == '__main__':
    try:
        losses = read_tensorboard_data('./logs')
        plot_losses(losses)
    except Exception as e:
        print(f"Error: {str(e)}")