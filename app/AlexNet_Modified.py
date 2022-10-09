import torch 
# import torchvision
import torch.functional as F
from torch import nn


class ImageClassifierBase(nn.Module):
  def training_step(self, batch):
    imgs, ids = batch
    out = self(imgs)
    loss = F.cross_entropy(out, ids)
    return loss
  
  def validation_step(self, batch):
    def acc(out, ids):
      _, preds = torch.max(out, dim=1)
      return torch.tensor(torch.sum(preds == ids).item() / len(preds))
    imgs, ids = batch
    out = self(imgs)
    loss = F.cross_entropy(out, ids)
    accuracy = acc(out, ids)
    return {"val_loss": loss, "val_acc": accuracy}
  
  def validation_epoch_end(self, outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean() 
    return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
  
  def print_epoch_stats(self, epoch, result):
    train_loss = result["train_loss"]
    val_loss = result["val_loss"]
    val_acc = result["val_acc"]
    print(f"Epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},\
          val_acc: {val_acc:.4f}")

class AlexNetModified(ImageClassifierBase):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 192, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, 192, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Flatten(),

        nn.Linear(192*3*3, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 10),
    )
  def forward(self, batch):
    return self.net(batch)