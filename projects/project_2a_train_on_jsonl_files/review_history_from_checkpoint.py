import torch
import sys
import argparse
import matplotlib.pyplot as plt
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')
import os
import pickle

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process and plot averaged loss history from a checkpoint.")
parser.add_argument('checkpoint_file', type=str, help="Path to the checkpoint file")
parser.add_argument('--average_over', type=int, default=1000, help="Number of iterations to average over (default: 1000)")
parser.add_argument('--output_file', type=str, default='averaged_loss_history.png', help="Output file for the plot (default: averaged_loss_history.png)")
args = parser.parse_args()

# Load the checkpoint
checkpoint_dict = torch.load(args.checkpoint_file, map_location=torch.device('cpu'), pickle_module=pickle)

# Extract and average loss history
loss_history = checkpoint_dict['loss_history'][1:]  # Skip the first element (0th iteration)
averaged_loss_history = [
    sum(loss_history[i:i + args.average_over]) / len(loss_history[i:i + args.average_over]) 
    for i in range(0, len(loss_history), args.average_over)
]

print(averaged_loss_history)

# Plot averaged loss history
plt.plot(range(len(averaged_loss_history)), averaged_loss_history)
plt.xlabel("Epochs (in chunks of averaged iterations)")
plt.ylabel("Negative Log-Likelihood (NLL)")
plt.title(f"Averaged Loss History (Averaging over {args.average_over} iterations)")

# Save the plot to the specified output file
output_path = os.path.join(args.output_file)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")