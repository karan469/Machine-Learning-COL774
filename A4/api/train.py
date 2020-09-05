import torch
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms
import PIL

from data_loader import get_loader,CocoDataset
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_RESIZE = (224, 224)

num_epochs=5
batch_size=64
num_workers=2
learning_rate=0.0008

num_layers=1
hidden_size=512
embed_size=512

log_step=10
crop_size=224
save_step=1000

img_transform = transforms.Compose([
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
                                    ])


data_loader = get_loader('../../../ml-a4-data/train_images/train_images', '../../../ml-a4-data/train_captions.tsv',
                             img_transform, batch_size,
                             shuffle=True, num_workers=num_workers)

print('data loaded')
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, data_loader.dataset.vocab.__len__(), num_layers).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
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

        if (i+1) % save_step == 0:
            torch.save(decoder.state_dict(),'decoder-{}-{}.ckpt'.format(epoch+1, i+1))
            torch.save(encoder.state_dict(),'encoder-{}-{}.ckpt'.format(epoch+1, i+1))

torch.save(decoder.state_dict(),'./decoder-final.ckpt')
torch.save(encoder.state_dict(),'./encoder-final.ckpt')
