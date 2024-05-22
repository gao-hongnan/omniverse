import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchvision.transforms import *

from .model import SimpleCNN
from .utils import *

train_dl, _ = get_dl(np.arange(2, 8), H5_PATH, test=False)

model = SimpleCNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

for epoch in range(EPOCHES):
    tot_loss = torch.zeros(1).to(device)
    bar = tqdm.tqdm(enumerate(train_dl))

    for n_batch, (x, y) in bar:
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)
        print(x.shape)
        mask = y != -1

        mask = mask.to(device).float()

        y_ = model(x)
        loss = criterion(y_ * mask, y * mask)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            tot_loss += loss

        bar.set_description(f"epoch={epoch}, loss={(tot_loss / (n_batch + 1)).cpu().item():.4f}")

torch.save(model, "./model.pt")
