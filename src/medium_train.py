import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import datetime

# Add the project directory to the path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Import your modules
from src.data.data_loader import HemorrhageDataset, get_data_loaders
from src.models.unet import UNet, combined_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    """
    Train the segmentation model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run the model on (cuda/cpu)
        num_epochs: Number of training epochs
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Initialize metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        # Use tqdm for a progress bar
        train_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in train_iter:
            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # Calculate Dice coefficient
            pred = torch.sigmoid(outputs)
            pred_binary = (pred > 0.5).float()
            dice = (2. * (pred_binary * masks).sum()) / (pred_binary.sum() + masks.sum() + 1e-8)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            train_dice += dice.item() * inputs.size(0)
            
            # Update progress bar
            train_iter.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_dice = train_dice / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch in val_iter:
                inputs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                # Calculate Dice coefficient
                pred = torch.sigmoid(outputs)
                pred_binary = (pred > 0.5).float()
                dice = (2. * (pred_binary * masks).sum()) / (pred_binary.sum() + masks.sum() + 1e-8)
                
                val_loss += loss.item() * inputs.size(0)
                val_dice += dice.item() * inputs.size(0)
                
                val_iter.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate epoch statistics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_dice = val_dice / len(val_loader.dataset)
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)
        
        # Print epoch results
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}')
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = Path('results/models')
            model_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_dir / f'unet_best_model_{timestamp}.pth')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation loss and Dice coefficient."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot Dice coefficient
    ax2.plot(epochs, history['train_dice'], 'b-', label='Training Dice')
    ax2.plot(epochs, history['val_dice'], 'r-', label='Validation Dice')
    ax2.set_title('Training and Validation Dice Coefficient')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Coefficient')
    ax2.legend()
    
    # Save the figure
    plt.tight_layout()
    fig.savefig('results/visualizations/training_history.png')
    plt.close(fig)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Define paths
    image_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    label_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and subset it
    print("Creating data loaders...")
    dataset = HemorrhageDataset(image_dir, label_dir, window_types=['brain_window'])
    
    # Use a subset of the data (300 samples)
    subset_size = 300
    print(f"Using {subset_size} samples out of {len(dataset)}")
    indices = torch.randperm(len(dataset))[:subset_size].tolist()
    
    # Split into training and validation
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders with reduced batch size
    batch_size = 4
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=2  # Reduced from 4
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2  # Reduced from 4
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create the model
    print("Initializing model...")
    model = UNet(n_channels=3, n_classes=1)
    
    # Define loss function and optimizer
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model with a moderate number of epochs
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10  # A moderate number of epochs
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()