import torch
import torch.optim as optim
import copy # For making deep copies of models, very important!
import time # To see how long things take

# --- 1. Import from Your Other Amazing Project Files ---
from generate_data import generate_all_synthetic_data
from model_setup import get_my_vgg16_fasterrcnn_model
from my_datasets import create_source_dataloader # We'll use this for each client's data

print("--- FedAvg Script Starting ---")

# --- 2. Configuration Settings ---
# These are like the knobs and dials for our experiment
NUM_CLIENTS = 2  # Let's start with 2 clients
NUM_COMMUNICATION_ROUNDS = 5 # How many times the server and clients will "talk"
LOCAL_EPOCHS_PER_CLIENT = 3  # How many times each client trains on its own data per round
CLIENT_BATCH_SIZE = 2        # Batch size for each client's local training (keep small!)
CLIENT_LEARNING_RATE = 0.001 # Learning rate for client's optimizer

# Data generation parameters (should match what your model expects)
TOTAL_SOURCE_IMAGES = 20 # Total number of source images to generate and split
NUM_MODEL_CLASSES = 2    # 1 object class + 1 background

# --- 3. Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 4. Generate Data and Distribute to Clients ---
print("\n--- Preparing Data for Clients ---")
# Generate all source data once (annotations and image paths)
all_source_annotations, _ = generate_all_synthetic_data(
    num_images_to_generate=TOTAL_SOURCE_IMAGES
)
print(f"Generated {len(all_source_annotations)} total source images with annotations.")

# Split the annotations among the clients
# This is a simple way to do it. Each client gets a chunk.
client_specific_annotations = []
images_per_client = len(all_source_annotations) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start_idx = i * images_per_client
    # Make sure the last client gets any remaining images if it doesn't divide perfectly
    end_idx = (i + 1) * images_per_client if i < NUM_CLIENTS - 1 else len(all_source_annotations)
    client_specific_annotations.append(all_source_annotations[start_idx:end_idx])

# Create a DataLoader for each client
client_dataloaders = []
for i in range(NUM_CLIENTS):
    if len(client_specific_annotations[i]) > 0:
        print(f"Creating DataLoader for Client {i+1} with {len(client_specific_annotations[i])} images.")
        client_dl = create_source_dataloader(
            ann_list=client_specific_annotations[i],
            batch_size_to_set=CLIENT_BATCH_SIZE,
            shuffle_data=True # Good to shuffle for training
        )
        client_dataloaders.append(client_dl)
    else:
        print(f"Client {i+1} has no data. Skipping DataLoader creation.")
        # You might want to handle this case, e.g., by ensuring all clients get data
        # or by not including this client in the round. For now, we assume all get some.

if len(client_dataloaders) != NUM_CLIENTS:
    print(f"Warning: Expected {NUM_CLIENTS} dataloaders, but created {len(client_dataloaders)}. Adjust data split or NUM_CLIENTS.")
    # For simplicity, we'll proceed if at least one client has data.

# --- 5. Initialize the Global Model (on the "Server") ---
print("\n--- Initializing Global Model ---")
global_model = get_my_vgg16_fasterrcnn_model(num_classes_to_set=NUM_MODEL_CLASSES)
global_model.to(device)
print("Global model initialized and moved to device.")

# --- 6. Client Update Function (What each client does locally) ---
def client_local_training(client_id, model_to_train, dataloader, epochs, learning_rate):
    """
    Simulates a single client's local training process.
    Args:
        client_id (int): Identifier for the client.
        model_to_train (torch.nn.Module): A deep copy of the global model.
        dataloader (DataLoader): The client's local data.
        epochs (int): Number of local epochs to train.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        dict: The state_dict of the locally trained model.
    """
    print(f"  Client {client_id+1}: Starting local training for {epochs} epochs...")
    model_to_train.train() # Set model to training mode
    
    # Setup optimizer for this client's local training
    optimizer = optim.SGD(model_to_train.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model_to_train(images, targets) # Forward pass
            losses = sum(loss for loss in loss_dict.values()) # Sum losses
            losses.backward() # Backward pass
            optimizer.step() # Update weights

            epoch_loss += losses.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"    Client {client_id+1}, Epoch {epoch+1}/{epochs}, Avg Batch Loss: {avg_epoch_loss:.4f}")

    print(f"  Client {client_id+1}: Finished local training.")
    return model_to_train.state_dict() # Return the learned weights

# --- 7. Server Aggregation Function (FedAvg Logic) ---
def federated_average_weights(list_of_client_state_dicts):
    """
    Averages the weights from a list of client model state_dicts.
    """
    if not list_of_client_state_dicts:
        return None

    print("  Server: Aggregating client model weights (FedAvg)...")
    # Initialize a new state_dict for the averaged weights, starting with zeros
    averaged_state_dict = copy.deepcopy(list_of_client_state_dicts[0]) # Use one as a template for keys and shapes
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = torch.zeros_like(averaged_state_dict[key])

    # Sum weights from all clients
    for client_state_dict in list_of_client_state_dicts:
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] += client_state_dict[key]

    # Divide by the number of clients to get the average
    num_clients = len(list_of_client_state_dicts)
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = torch.div(averaged_state_dict[key], num_clients)
    
    print("  Server: Weight aggregation complete.")
    return averaged_state_dict

# --- 8. Main Federated Learning Loop ---
print(f"\n--- Starting Federated Averaging for {NUM_COMMUNICATION_ROUNDS} Communication Rounds ---")

for comm_round in range(NUM_COMMUNICATION_ROUNDS):
    round_start_time = time.time()
    print(f"\nCommunication Round {comm_round + 1}/{NUM_COMMUNICATION_ROUNDS}")

    # List to store the state_dicts from clients after their local training this round
    client_updated_state_dicts = []

    # Simulate clients training in parallel (we'll do it sequentially here)
    for client_idx in range(NUM_CLIENTS):
        if client_idx < len(client_dataloaders): # Check if this client has data
            print(f" Processing Client {client_idx+1}/{NUM_CLIENTS}...")
            
            # IMPORTANT: Give each client a DEEP COPY of the current global model
            # This ensures that clients train independently and don't modify the same model object.
            local_model_copy = copy.deepcopy(global_model)
            local_model_copy.to(device) # Ensure the copy is on the correct device
            
            # Client performs local training
            updated_state_dict = client_local_training(
                client_id=client_idx,
                model_to_train=local_model_copy,
                dataloader=client_dataloaders[client_idx],
                epochs=LOCAL_EPOCHS_PER_CLIENT,
                learning_rate=CLIENT_LEARNING_RATE
            )
            client_updated_state_dicts.append(updated_state_dict)
            
            # Clean up the copied model to save memory (optional, but good practice in simulations)
            del local_model_copy
            if device.type == 'cuda':
                torch.cuda.empty_cache() 
        else:
            print(f" Skipping Client {client_idx+1} as it has no dataloader.")


    # Server aggregates the updated model weights from clients
    if client_updated_state_dicts:
        new_global_model_state_dict = federated_average_weights(client_updated_state_dicts)
        global_model.load_state_dict(new_global_model_state_dict) # Update the global model
        print("  Server: Global model has been updated with averaged weights.")
    else:
        print("  Server: No client updates received this round. Global model remains unchanged.")

    round_duration = time.time() - round_start_time
    print(f"Communication Round {comm_round + 1} finished in {round_duration:.2f}s.")
    
    # Optional: You could evaluate the global_model on a test set here periodically
    # to see how its performance improves over rounds.

print("\n--- Federated Averaging Process Finished ---")

# Optional: Save the final trained global model
# final_global_model_path = 'fedavg_final_global_model.pth'
# torch.save(global_model.state_dict(), final_global_model_path)
# print(f"Final FedAvg global model saved to {final_global_model_path}")

