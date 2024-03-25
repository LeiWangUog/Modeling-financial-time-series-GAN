import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

csv_file_path = '/content/BABL_data_manager.csv'
data = pd.read_csv(csv_file_path)
Log_Return_data = data[['Log Return']]
sequence_length = 8400
import numpy as np

def create_sequences(data, sequence_length):
    xs = []
    data_values = data['Log Return'].values
    for i in range(len(data_values) - sequence_length):
        # 从数据中抽取长度为sequence_length的序列
        x = data_values[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)


sequence_length = 8400  
training_data = create_sequences(Log_Return_data, sequence_length)





training_data_tensor = torch.from_numpy(training_data).float()


batch_size = 70
train_dataset = TensorDataset(training_data_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



losses_G = []
losses_D = []
d_real_outputs = []
d_fake_outputs = []

criterion = nn.BCELoss()

num_epochs = 500

for epoch in range(num_epochs):
    for real_data_samples, in train_loader:
 
        N = real_data_samples.size(0)  # 获取当前批次的真实大小
        real_data_samples = real_data_samples.view(N, 1, sequence_length)  # 使用N而不是batch_size

      
        real_labels = torch.ones(N, 1, device=device)
        fake_labels = torch.zeros(N, 1, device=device)
        real_data_samples = real_data_samples.to(device).view(N, 1, -1)
        optimizer_D.zero_grad()
        output_real = discriminator(real_data_samples)
        loss_real = criterion(output_real, real_labels).to(device)

        noise = torch.randn(N, input_size, device=device)
        fake_data = generator(noise)
        fake_data = fake_data.view(N, 1, -1)
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

      
        optimizer_G.zero_grad()
        output = discriminator(fake_data)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizer_G.step()
      
        losses_G.append(loss_G.item())
        losses_D.append(loss_D.item())
        d_real_outputs.append(output_real.mean().item())
        d_fake_outputs.append(output_fake.mean().item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}')
