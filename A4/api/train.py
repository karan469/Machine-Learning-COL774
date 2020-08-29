import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader,Rescale,ToTensor,CocoDataset
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_RESIZE = (128, 128)
batch_size = 128
num_workers=2

num_epochs=5
batch_size=128
num_workers=2
learning_rate=0.001
num_layers=1
hidden_size=512
embed_size=256

log_step=10
crop_size=128

save_step=1000

transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor()])


img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor()])
data_loader = get_loader('../../../ml-a4-data/train_images/train_images', '../../../ml-a4-data/train_captions.tsv', 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers)

# vocab = CocoDataset(root='../../../ml-a4-data/train_images/train_images',
#                        captions_file_path='../../../ml-a4-data/train_captions.tsv',
#                        img_transform=img_transform).vocab

# print(len(vocab))

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, 5446, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate) 

# Train the models
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
        
        # Set mini-batch dataset
        images = images.to(device).float()
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        print(targets)

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
            
        # Save the model checkpoints
        if (i+1) % save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                './', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                './', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))