import torch
import time
import copy # Für das Deep Copy der besten Gewichte

def run_training(volume_provider, trainer, epochs=2000, batch_size=1024, 
                 sparse_factor=1, save_path="brain_scene.pth", 
                 early_stop_threshold=1e-7):
    """
    Orchestrates the training and saves the BEST model weights.
    """
    
    total_slices = volume_provider.get_total_slices()
    train_indices = list(range(0, total_slices, sparse_factor))
    
    print(f"Starting Training: {len(train_indices)}/{total_slices} slices (Factor: {sparse_factor})")
    
    # --- BEST MODEL TRACKING ---
    best_loss = float('inf')
    best_model_state = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for slice_idx in train_indices:
            slice_2d, metadata = volume_provider.get_slice(axis='z', index=slice_idx)
            loss = trainer.train_step(slice_2d, metadata, batch_size=batch_size)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_indices)

        # --- CHECKPOINT LOGIC: Save if this is the best loss so far ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Wir speichern eine Kopie der Gewichte im RAM
            best_model_state = copy.deepcopy(trainer.model.state_dict())

        # --- EARLY STOPPING ---
        if avg_loss < early_stop_threshold:
            print(f"\n[EARLY STOP] Epoch {epoch+1}: Loss {avg_loss:.8f} < Threshold {early_stop_threshold}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.8f} | Best: {best_loss:.8f} | {elapsed:.1f}s")

    # --- FINAL SAVE ---
    # Wir laden die besten gefundenen Gewichte zurück ins Modell, bevor wir speichern
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state)
        torch.save(best_model_state, save_path)
        print(f"Training complete. Best model saved to {save_path} (Loss: {best_loss:.8f})")
    else:
        torch.save(trainer.model.state_dict(), save_path)
        print(f"Training complete. Last model saved to {save_path}")
    
    return trainer.model