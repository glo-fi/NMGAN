import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import random
from scipy.io import wavfile
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import truncated_normal, orthogonal_regularization

from models_deeper_8s import Discriminator, Generator, SonataDataset

hq_train_folder = r'data/sonatas_8s'
lq_train_folder = r'data/sonatas_lq_8s'
hq_test_folder = r'data/sonatas_minitest_8s'
lq_test_folder = r'data/sonatas_lq_minitest_8s'
serialized_train_folder = r'data/sonatas_serial_8s'
serialized_test_folder = r'data/sonatas_serial_minitest_8s'

if __name__ == '__main__':
    device = torch.device("cuda:0,1,2,3,4,5" if torch.cuda.is_available() else "cpu")
    print("\nNo. of GPUs: ", torch.cuda.device_count())
    discriminator = Discriminator()
    #discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count())) # use multiple GPUs
    discriminator.to(device)
    discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))
    print(discriminator)
    print('Discriminator created')

    generator = Generator()
   #generator = nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
    generator.to(device)
    generator = nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
    print(generator)
    print('Generator created')


    g_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, weight_decay = 0.0001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr = 0.00075, weight_decay = 0.0001)

    BATCH_SIZE = 25
    
    NUM_EPOCHS = 80

    print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
    torch.cuda.empty_cache()
    print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))
    

    # load data
    print('loading data...')
    train_dataset = SonataDataset(data_type='train')
    test_dataset = SonataDataset(data_type='test')
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))

for epoch in range(NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_hq, train_lq in train_bar:
            torch.cuda.empty_cache() 
            # latent vector - normal distrbution
            z = truncated_normal((train_batch.size(0), 1024, 16)) #using a truncated normal
            if torch.cuda.is_available():
                train_batch, train_hq, train_lq = train_batch.to(device), train_hq.to(device), train_lq.to(device)
                z = z.to(device)
            train_batch, train_hq, train_lq = Variable(train_batch), Variable(train_hq), Variable(train_lq)
            z = Variable(z)
 
            # TRAIN D to recognize hq audio as hq

            # training batch pass

            discriminator.zero_grad()
            outputs = discriminator(train_batch)
            #label = random.uniform(0.8, 1.01) # random label switching
            #if 1.0 < label:
            #    label = 0
            hq_loss = torch.mean((outputs - label) ** 2)  # L2 loss - we want them all to be 1
            hq_loss.backward()



            # TRAIN D to recognize generated audio as lq
            
            generated_outputs = generator(train_lq, z)
            outputs = discriminator(torch.cat((generated_outputs, train_lq), dim=1))
            #label = random.uniform(0.0, 0.21) #random label switching
            #if 0.2 < label:
            #    label = 1
            lq_loss = torch.mean((outputs - label) ** 2)  # L2 loss
            lq_loss.backward()

            d_optimizer.step()  # update parameters


            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_lq, z)
            gen_lq_pair = torch.cat((generated_outputs, train_lq), dim=1)
            outputs = discriminator(gen_lq_pair)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)

            penalty = orthogonal_regularization(generator, device) #orthogonal regularisation
	    l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_hq)))
            g_cond_loss = 50 * torch.mean(l1_dist)  # conditional loss
     
            lambda_penalty = 1e-5 * penalty
            g_loss = g_loss_ + g_cond_loss + lambda_penalty

            


            # backprop, optimize

            g_loss.backward()
            g_optimizer.step()

            train_bar.set_description(
                'Epoch {}: d_hq_loss {:.4f}, d_lq_loss {:.4f}, g_loss {:.4f}, penalty {:.4f}'
                    .format(epoch + 1, hq_loss.data, lq_loss.data, g_loss.data, lambda_penalty.data)
)


        # TEST model

        output_folder = r'results_8s_mini_deep_specnorm3'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_lq in test_bar:
            z = truncated_normal((test_lq.size(0), 1024, 16))
            if torch.cuda.is_available():
                test_lq, z = test_lq.to(device), z.to(device)
            test_lq, z = Variable(test_lq), Variable(z)
            fake_beethoven = generator(test_lq, z).data.cpu().numpy()  # convert to numpy array
            
            for idx in range(fake_beethoven.shape[0]):
                generated_sample = fake_beethoven[idx]
                
                file_name = os.path.join(output_folder,
                                         '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
                wavfile.write(file_name, 8000, generated_sample.T)
     



        # save the model parameters for each epoch -- this really isn't necessary as saved model size is very large and saving 100 different models can easily take up dozens of gigabytes.
        epoch_file = r'epochs_8s_deep_specnorm3'
        if not os.path.exists(epoch_file):
            os.makedirs(epoch_file)
        
        g_path = os.path.join(epoch_file, 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join(epoch_file, 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
        torch.cuda.empty_cache()
