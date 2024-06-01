from scripts.model import NTXent
import torch
import math
import wandb
from scripts.utils import save_model

def train_encoder(train_df,
                DataFrameToDataLoader, 
                model, 
                device, 
                target_col,
                batch_size,
                lr,
                epochs,
                model_save_dir,
                model_name):
    """
    Train a given model using the provided data and configurations.
    """
    # Convert DataFrames to DataLoader
    train_dl = DataFrameToDataLoader(train_df, target_col=target_col, batch_size=batch_size).dataloader

    loss_func = NTXent(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / batch_size)

    best_train_loss = float('inf')  # Initializing with a high value 

    # Training loop
    for epoch in range(epochs):
        model.train()

        for step, (anchor, positive) in enumerate(train_dl):
            anchor, positive = anchor.to(device), positive.to(device)
            optimizer.zero_grad()
            emb_anchor, emb_positive = model(anchor, positive)
            train_loss = loss_func(emb_anchor, emb_positive)
            train_loss.backward()
            optimizer.step()


            train_metrics = {
                "encoder_train/train_loss": train_loss.item(),
            }

            if step + 1 < n_steps_per_epoch:
                wandb.log(train_metrics)  # Logging to wandb

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.3f}")

        if best_train_loss > train_loss:
            best_train_loss = train_loss
            save_model(model, model_dir=model_save_dir, model_name=model_name)