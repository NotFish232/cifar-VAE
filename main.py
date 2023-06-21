import torch as T
from torch import optim, nn
from dataset import CIFARDataset
from torch.utils.data import DataLoader
from network import VAE
from tqdm import tqdm
import imageio
from torchvision.transforms import Compose, Lambda
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

def kl_divergence(mean: T.Tensor, logvar: T.Tensor) -> T.Tensor:
    return -0.5 * T.sum(1 + logvar - mean ** 2 - logvar.exp())
    


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    transforms = Compose([
        Lambda(lambda x: x.reshape(3, 32, 32) / 255),
        Lambda(lambda x: T.tensor(x, device=device, dtype=T.float32))
    ])
    dataset = CIFARDataset(transforms=transforms)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    network = VAE().to(device)
    optimizer = optim.Adam(network.parameters(), LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 100, eta_min=1e-8)

    criterion = nn.MSELoss()

    z = T.randn(256, device=device)
    frames = []

    for epoch in range(1, EPOCHS + 1):
        acc_loss = 0
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            y, (mean, logvar) = network(imgs)
            loss = criterion(y, imgs) + 0.1 * kl_divergence(mean, logvar)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss.item()
        
            scheduler.step()
        
        print(f"Loss: {acc_loss: .2f}")

        with T.no_grad():
            frame = network.decoder(z)[0]
            frames.append((255 * frame.permute(1, 2, 0)).to(T.uint8).cpu().numpy())
    
    T.save(network.state_dict(), "trained_model.pt")

    imageio.mimsave("progress.gif", frames, duration=EPOCHS)


if __name__ == "__main__":
    main()