import time
import torch

from collections import defaultdict
from sklearn.metrics import accuracy_score


def training_loop(
    model, optimizer, train_dataloader, val_dataloader, n_epochs, device=None
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Training device: {device}")
    
    model = model.to(device)

    history = defaultdict(list)
    for i_epoch in range(1, n_epochs + 1):
        t0 = time.time()
        loss_sum = 0
        accuracy_sum = 0
        model.train()
        for (
            seqs,
            pitches,
            keysignatures,
            lens,
        ) in train_dataloader:  # seqs, pitches, keysignatures, lens are batches
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

            with torch.no_grad():
                predicted_pitch, predicted_ks = model.predict(seqs, lens)
                for i, p in enumerate(predicted_pitch):
                    # compute the accuracy without considering the padding
                    acc = accuracy_score(p, pitches[:, i][: len(p)].cpu())
                    # normalize according to the number of sequences in the batch
                    accuracy_sum += acc / len(lens)

        train_loss = loss_sum / len(train_dataloader)
        # normalize according to the number of batches
        train_accuracy = accuracy_sum / len(train_dataloader)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Evaluate on the validation set
        model.eval()
        all_predicted_pitch = []
        all_predicted_ks = []
        all_pitches = []
        all_ks = []
        with torch.no_grad():
            for seqs, pitches, keysignatures, lens in val_dataloader:
                # Predict the model's output on a batch
                predicted_pitch, predicted_ks = model.predict(seqs.to(device), lens)
                # Update the lists that will be used to compute the accuracy
                for i, (p, k) in enumerate(zip(predicted_pitch, predicted_ks)):
                    all_predicted_pitch.append(torch.Tensor(p))
                    all_predicted_ks.append(torch.Tensor(k))
                    all_pitches.append(pitches[0 : int(lens[i]), i])
                    all_ks.append(keysignatures[0 : int(lens[i]), i])

        # Compute the overall accuracy for the validation set
        val_accuracy_pitch = accuracy_score(
            torch.cat(all_predicted_pitch), torch.cat(all_pitches)
        )
        val_accuracy_ks = accuracy_score(torch.cat(all_predicted_ks), torch.cat(all_ks))
        history["val_accuracy_pitch"].append(val_accuracy_pitch)
        history["val_accuracy_ks"].append(val_accuracy_ks)

        #         save the model
        torch.save(model, "./models/temp/model_temp_epoch{}.pkl".format(i_epoch))
        #         files.download("model_temp_epoch{}.pkl".format(i_epoch))

        t1 = time.time()
        print(
            f"Epoch {i_epoch}: train loss = {train_loss:.4f}, train_accuracy: {train_accuracy:.4f},val_accuracy_pitch: {val_accuracy_pitch:.4f},val_accuracy_ks: {val_accuracy_ks:.4f}, time = {t1-t0:.4f}"
        )
    return history