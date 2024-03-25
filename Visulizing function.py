import pandas as pd
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(d_real_outputs, label='Real: D(x)')
plt.plot(d_fake_outputs, label='Fake: D(G(z))')
plt.xlabel('Epoch')
plt.ylabel('Discriminator output')
plt.title('Discriminator Outputs During Training')
plt.legend()
plt.show()


#Data Processing
all_fake_data = []
fake_data_2d = fake_data.squeeze(1).detach() 
all_fake_data.append(fake_data_2d)
concatenated_data_2d = torch.cat(all_fake_data, dim=0)  
flat_data = concatenated_data_2d.view(-1).cpu().numpy()  

csv_file_path = 'generated_log_returns.csv'
pd.DataFrame(flat_data).to_csv(csv_file_path, index=False)

## Volatility Clustering_real_BABL##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_acf(x, max_lag=1000, for_abs=True):
    if for_abs:
        x = np.abs(x)
    res = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        res[lag - 1] = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return res

def visualize_acf(acf_values, file_name, scale='log'):
    plt.figure(dpi=150)
    plt.plot(np.linspace(1, len(acf_values), len(acf_values)), acf_values, '.')
    plt.ylim(1e-5, 1.)
    if scale == 'linear':
        plt.ylim(-1., 1.)
    plt.xscale('log')
    plt.yscale(scale)
    plt.xlabel('lag k', fontsize=10)
    plt.ylabel('Auto-correlation', fontsize=10)
    plt.title("Volatility Clustering on BABL", fontsize=10)  
    plt.savefig(file_name + '.png', transparent=True)
    plt.close()


generated_data_BABL = pd.read_csv('M:\Machine Learing\BABL\BABL_data_manager.csv')
log_return = generated_data_BABL['Log Return']  

acf_values = calculate_acf(log_return, max_lag=1000, for_abs=True)  

file_name = "Volatility Clustering_real_BABL"  
visualize_acf(acf_values, file_name, scale='log')  

#coarse_volatility


file_path = 'M:\Machine Learing\BABL\BABL_Fake_colab_1744.csv'
generated_data_BABL = pd.read_csv(file_path, header=None)#nrows=8190


generated_data_BABL.columns = ['Log Return']
log_return = generated_data_BABL['Log Return']

tau = 5

coarse_volatility = np.abs(log_return).rolling(window=tau).sum()

fine_volatility = log_return.abs().rolling(window=1).sum()

lead_lag_correlation = lambda k: coarse_volatility.corr(fine_volatility.shift(k))

correlation_at_lag_1 = lead_lag_correlation(1)

correlation_difference = lead_lag_correlation(1) - lead_lag_correlation(-1)


lags = range(-20, 21)  
correlations = [lead_lag_correlation(k) for k in lags]

asymmetries = [lead_lag_correlation(k) - lead_lag_correlation(-k) for k in range(1, 21)]

#plotting
plt.figure(figsize=(10, 5))
plt.plot(lags, correlations, 'bo', label='Coarse-Fine Correlation')
plt.plot(range(1, 21), asymmetries, 'ro-', label='Asymmetry')
plt.axhline(y=0, color='k', linestyle='--')  
plt.xlabel('Lag k')
plt.ylabel('Correlation')
plt.title('Lead-Lag Correlation of Coarse-Fine Volatility')
plt.legend()
plt.show()

#Gain and loss asymetary 


file_path = 'M:\Machine Learing\BABL\BABL_data_manager.csv'
generated_data_BABL = pd.read_csv(file_path)



log_return = generated_data_BABL['Log Return']


theta_pos = 0.1
theta_neg = -0.1




wait_times_positive = []
wait_times_negative = []


for t in range(len(log_return) - 1):

    for future_t in range(t + 1, len(log_return)):
        if log_return[future_t] - log_return[t] >= theta_pos:
            wait_times_positive.append(future_t - t)
            break

    for future_t in range(t + 1, len(log_return)):
        if log_return[future_t] - log_return[t] <= theta_neg:
            wait_times_negative.append(future_t - t)
            break

max_wait = max(max(wait_times_positive), max(wait_times_negative))
wait_times_distribution_pos = [wait_times_positive.count(i) / len(wait_times_positive) for i in range(1, max_wait + 1)]
wait_times_distribution_neg = [wait_times_negative.count(i) / len(wait_times_negative) for i in range(1, max_wait + 1)]

plt.figure(figsize=(10, 6))  


plt.scatter(range(1, max_wait + 1), wait_times_distribution_pos, color='red', s=10, label='Positive threshold', alpha=0.6)
plt.scatter(range(1, max_wait + 1), wait_times_distribution_neg, color='blue', s=10, label='Negative threshold', alpha=0.6)

plt.xscale('log')
plt.yscale('linear')

plt.xlabel('Time ticks t\'')
plt.ylabel('Return time probability')
plt.title('Gain/Loss Asymmetry')
plt.legend()
plt.show()
