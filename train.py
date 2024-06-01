from scripts.scarf import NTXent
import torch
import math
import wandb
from torch import nn
from scripts.utils import save_model, save_wandb_metrics,compute_metrics
from torch.nn.utils import clip_grad_norm_

def log_input_table(inputs, predicted, labels, probs):
    """
    Log a wandb.Table with inputs, predictions, true labels, and class probabilities.
    """
    num_classes = probs.shape[1]
    
    # Create a wandb Table to log inputs, labels, and predictions to
    table = wandb.Table(columns=["input", "pred", "target"] + [f"score_{i}" for i in range(num_classes)])
    
    for input_data, pred, targ, prob in zip(inputs.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(str(input_data.numpy()), pred.item(), targ.item(), *prob.numpy())
    
    wandb.log({"predictions_table": table}, commit=False)


def validate_model(model, valid_dl, loss_func, device, energyRatingEncoding, log_inputs=False, batch_idx=0):
    """
    Compute performance of the model on the validation dataset 
    and optionally log a wandb.Table of inputs, predictions, and labels.
    """
    model.eval()
    val_loss = 0.
    
    # Variables to store the per-category metrics
    all_labels = []
    all_outputs = []

    with torch.inference_mode():
        for i, (inputs, labels) in enumerate(valid_dl):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            if loss_func:
                val_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Append outputs and labels for later accuracy calculation
            all_outputs.append(outputs)
            all_labels.append(labels)

            _, predicted = torch.max(outputs.data, 1)

            # Log one batch of inputs to W&B, always same batch_idx.
            if i == batch_idx and log_inputs:
                log_input_table(inputs, predicted, labels, outputs.softmax(dim=1))

    # Concatenate all outputs and labels from the batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # # Get the class names
    # class_names = valid_dl.dataset.classes

    # Compute overall and per-category accuracy using our function
    overall_accuracy, per_category_accuracy, f1, predictions, labels = compute_metrics(all_outputs, all_labels, energyRatingEncoding)
    if loss_func:            
        return val_loss / len(valid_dl.dataset), overall_accuracy, per_category_accuracy, f1, predictions, labels
    else:
        return overall_accuracy, per_category_accuracy, f1, predictions, labels

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

def train_mlp(train_df, test_df, 
                DataFrameToDataLoader, 
                model, 
                device, 
                energyRatingEncoding, 
                target_col, 
                batch_size,
                lr,
                epochs,
                model_save_dir,
                model_name,
                early_stopping_threshold = 5):
    """
    Train a given model using the provided data and configurations.
    """
    # Convert DataFrames to DataLoader
    train_dl = DataFrameToDataLoader(train_df, target_col=target_col, batch_size=batch_size).dataloader
    test_dl = DataFrameToDataLoader(test_df, target_col=target_col, batch_size=batch_size).dataloader

    # Loss and optimizer setup
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / batch_size)

    best_val_acc = -float('inf')  # Initializing with a high value
    epochs_without_improvement = 0
    early_stopping_threshold = early_stopping_threshold  

    # Training loop
    for epoch in range(epochs):
        model.train()

        for step, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_loss = loss_func(outputs, labels)

            train_overall_accuracy, _, train_f1, _, _ = compute_metrics(outputs, labels, energyRatingEncoding)

            optimizer.zero_grad()
            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            train_metrics = {
                "train/train_loss": train_loss.item(),
                "train/train_acc": train_overall_accuracy,
                "train/f1": train_f1
            }

            if step + 1 < n_steps_per_epoch:
                wandb.log(train_metrics)  # Logging to wandb

        # Validate
        test_loss, test_accuracy, test_category_accuracies, test_f1, _, _ = validate_model(
            model, test_dl, loss_func, device, energyRatingEncoding, log_inputs=(epoch == (epochs - 1))
        )

        # Early stopping logic
        if test_accuracy > best_val_acc:
            best_val_acc = test_accuracy
            epochs_without_improvement = 0
            # Optionally save the best model here
            save_model(model, model_dir=model_save_dir, model_name=model_name)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_threshold:
            print("Early stopping due to no improvement in validation loss.")
            break  # Stop the training loop


        val_metrics = {
            "test/test_loss": test_loss,
            "test/test_f1": test_f1,
            **save_wandb_metrics('test', test_accuracy, test_category_accuracies)
        }
        wandb.log(val_metrics)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.3f}, "
            f"Train Acc: {train_overall_accuracy:.3f}, "
            f"Train F1: {train_f1:.3f}, "
            f"Test Loss: {test_loss:.3f}, "
            f"Test Acc: {test_accuracy:.3f}, "
            f"Test F1: {test_f1:.3f}")


    # # Save model
    # save_model(model, model_dir=model_save_dir, model_name=model_name)

    # Test the model
    _, test_accuracy, test_category_accuracies, test_f1, predictions, true_labels = validate_model(
        model, test_dl, loss_func, device, energyRatingEncoding, log_inputs=(epoch == (epochs - 1))
    )
    
    # test_metrics = {
    #     "test/test_f1": test_f1,
    #     **save_wandb_metrics("test", test_accuracy, test_category_accuracies)
    # }
    # wandb.log(test_metrics)



    wandb.sklearn.plot_confusion_matrix(true_labels, predictions, labels=list(energyRatingEncoding.keys()))

    print(f"Test Accuracy: {test_accuracy:.3f}, Test F1: {test_f1:.3f}")

