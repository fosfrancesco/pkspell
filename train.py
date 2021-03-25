import time
import torch

from collections import defaultdict
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


def training_loop(
    model,
    optimizer,
    train_dataloader,
    epochs=50,
    val_dataloader=None,
    device=None,
    scheduler=None,
    writer=None,
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Training on device: {device}")

    if writer is None:
        writer = SummaryWriter(flush_secs=20)

    # Move model to GPU if needed
    model = model.to(device)

    history = defaultdict(list)
    for i_epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_sum = 0
        accuracy_pitch_sum = 0
        accuracy_ks_sum = 0
        model.train()
        for idx, (seqs, pitches, keysignatures, lens,) in enumerate(
            train_dataloader
        ):  # seqs, pitches, keysignatures, lens are batches
            seqs, pitches, keysignatures = (
                seqs.to(device),
                pitches.to(device),
                keysignatures.to(device),
            )
            optimizer.zero_grad()
            loss = model(seqs, pitches, keysignatures, lens)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            writer.add_scalar(
                "Loss/train", loss.item(), (i_epoch-1) * len(train_dataloader) + idx
            )

            with torch.no_grad():
                predicted_pitch, predicted_ks = model.predict(seqs, lens)
                for i, (p, k) in enumerate(zip(predicted_pitch, predicted_ks)):
                    # compute the accuracy without considering the padding
                    acc_pitch = accuracy_score(p, pitches[:, i][: len(p)].cpu())
                    acc_ks = accuracy_score(k, keysignatures[:, i][: len(k)].cpu())
                    # normalize according to the number of sequences in the batch
                    accuracy_pitch_sum += acc_pitch / len(lens)
                    accuracy_ks_sum += acc_ks / len(lens)

        train_loss = loss_sum / len(train_dataloader)
        # normalize according to the number of batches
        train_accuracy_pitch = accuracy_pitch_sum / len(train_dataloader)
        train_accuracy_ks = accuracy_ks_sum / len(train_dataloader)
        history["train_loss"].append(train_loss)
        history["train_accuracy_pitch"].append(train_accuracy_pitch)
        history["train_accuracy_ks"].append(train_accuracy_ks)
        writer.add_scalar("Accuracy_Pitch/train", train_accuracy_pitch, i_epoch)
        writer.add_scalar("Accuracy_KS/train", train_accuracy_ks, i_epoch)

        if val_dataloader is not None:
            # Evaluate on the validation set
            model.eval()
            accuracy_pitch_sum = 0
            accuracy_ks_sum = 0
            with torch.no_grad():
                for seqs, pitches, keysignatures, lens in val_dataloader:
                    # Predict the model's output on a batch
                    predicted_pitch, predicted_ks = model.predict(seqs.to(device), lens)
                    # Update the lists that will be used to compute the accuracy
                    for i, (p, k) in enumerate(zip(predicted_pitch, predicted_ks)):
                        # compute the accuracy without considering the padding
                        acc_pitch = accuracy_score(p, pitches[:, i][: len(p)].cpu())
                        acc_ks = accuracy_score(k, keysignatures[:, i][: len(k)].cpu())
                        # normalize according to the number of sequences in the batch
                        accuracy_pitch_sum += acc_pitch / len(lens)
                        accuracy_ks_sum += acc_ks / len(lens)
                
                # normalize according to the number of batches
                val_accuracy_pitch = accuracy_pitch_sum / len(val_dataloader)
                val_accuracy_ks = accuracy_ks_sum / len(val_dataloader)

            history["val_accuracy_pitch"].append(val_accuracy_pitch)
            history["val_accuracy_ks"].append(val_accuracy_ks)

            writer.add_scalar("Accuracy_Pitch/val", val_accuracy_pitch, i_epoch)
            writer.add_scalar("Accuracy_KS/val", val_accuracy_ks, i_epoch)
            if scheduler is not None:
                scheduler.step()
            #if scheduler is not None:
            #    scheduler.step(val_accuracy_pitch)

        #         save the model
        torch.save(model, "./models/temp/model_temp_epoch{}.pkl".format(i_epoch))
        #         files.download("model_temp_epoch{}.pkl".format(i_epoch))

        t1 = time.time()
        print(
            f"Epoch {i_epoch}: train loss = {train_loss:.4f}, train_accuracy_pitch: {train_accuracy_pitch:.4f}, train_accuracy_ks: {train_accuracy_ks:.4f},val_accuracy_pitch: {val_accuracy_pitch:.4f},val_accuracy_ks: {val_accuracy_ks:.4f}, time = {t1-t0:.4f}"
        )
    return history
