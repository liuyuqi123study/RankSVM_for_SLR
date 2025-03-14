import torch

def ranknet_loss(s_i, s_j, y_ij):
    """
    s_i: Score for document i
    s_j: Score for document j
    y_ij: Ground truth label (1 if i > j, -1 if i < j, 0 if i == j)
    """
    s_diff = s_i - s_j
    loss = torch.log(1 + torch.exp(-y_ij * s_diff))
    return loss.mean()

def train_ranknet(model,dataloader,optimizer,epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_i, x_j, y_ij = batch
            optimizer.zero_grad()

            # Forward pass
            s_i = model(x_i)
            s_j = model(x_j)

            # Compute loss
            loss = ranknet_loss(s_i, s_j, y_ij)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")