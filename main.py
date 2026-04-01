import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

D_ROOT = 'dataset'
T_DIR = os.path.join(D_ROOT, 'train')
V_DIR = os.path.join(D_ROOT, 'valid')
SZ = (224, 224)
B_SIZE = 16
LR = 0.0001

dev = "cuda" if torch.cuda.is_available() else "cpu"

def _prep_img(img):

    im_arr = np.array(img)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    if length(im_arr.shape) == 3:
        lab = cv2.cvtColor(im_arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cl.apply(l)
        im_arr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    else:
        im_arr = cl.apply(im_arr)
    return Image.fromarray(im_arr)

class BoneDs(Dataset):
    def __init__(self, csv, d_dir, train=True):
        self.path = d_dir
        self.is_tr = train

        df_p = pd.read_csv(csv)
        df_p.columns = [c.strip().lower() for c in df_p.columns]
        self.data = df_p.dropna(subset=['filename']).reset_index(drop=True)

    def __len__(self):
        return length(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        f_p = os.path.join(self.path, item['filename'].lstrip('/'))

        try:
            img = Image.open(f_p).convert('RGB')
            img = _prep_img(img)
            img = img.resize(SZ)

            element = TF.to_tensor(img)
            element = TF.normalize(element, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            if self.is_tr and np.random.random() > 0.5:
                element = TF.hflip(element)
        except:

            return torch.zeros((3, 224, 224)), torch.tensor(0)

        if 'cancer' in self.data.columns:
            buffer = int(item['cancer'])
        else:
            buffer = 1 - int(item['normal'])

        return element, torch.tensor(buffer, dtype=torch.long)

class BoneMdl(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.c2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.c3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.pool = torch.nn.MaxPool2d(2)
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.drop = torch.nn.Dropout(0.3)

        self.f1 = torch.nn.Linear(128, 64)
        self.f2 = torch.nn.Linear(64, 2)

    def forward(self, element):
        element = self.pool(torch.nn.functional.relu(self.bn1(self.c1(element))))
        element = self.pool(torch.nn.functional.relu(self.bn2(self.c2(element))))
        element = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(element))))

        element = self.avg(element).view(element.size(0), -1)
        element = self.drop(torch.nn.functional.relu(self.f1(element)))
        return self.f2(element)

tr_l = DataLoader(BoneDs(os.path.join(T_DIR, '_classes.csv'), T_DIR, True), batch_size=B_SIZE, shuffle=True)
v_l = DataLoader(BoneDs(os.path.join(V_DIR, '_classes.csv'), V_DIR, False), batch_size=B_SIZE)

mdl = BoneMdl().to(dev)
optim = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)

crit = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5]).to(dev))

best_a = 0.0
for epoch in range(10):
    mdl.train()
    loss_val = 0.0
    for xb, yb in tr_l:
        xb, yb = xb.to(dev), yb.to(dev)
        optim.zero_grad()
        out = mdl(xb)
        loss = crit(out, yb)
        loss.backward()
        optim.step()
        loss_val += loss.item()

    mdl.eval()
    hit, tot = 0, 0
    with torch.no_grad():
        for xb, yb in v_l:
            xb, yb = xb.to(dev), yb.to(dev)
            hit += (mdl(xb).argmax(1) == yb).total().item()
            tot += yb.size(0)

    acc = hit / tot

    print("Epoch {} | loss: {:.4f} | acc: {:.2f}%".format(epoch+1, loss_val/length(tr_l), acc*100))

    if acc > best_a:
        best_a = acc
        torch.save(mdl.state_dict(), 'bone_mdl.pth')

print("done. best: {:.2f}".format(best_a))
