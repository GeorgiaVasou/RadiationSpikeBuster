import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_cleaning as dc

filename = "C:/Users/vasou/Desktop/PYTHON SAT/dataset_ace_epam_rt.csv"

data = np.genfromtxt(fname=filename, delimiter=',', skip_header=1)

t = np.datetime64("0000-01-01") + data[:, 0] * np.timedelta64(1, 'ms')

# Create a DataFrame for the data
x = {'x1': data[:, 3], 'x2': data[:, 4], 'x_quality': data[:, 2]}
df = pd.DataFrame(x)

# Set x1 and x2 to NaN where x_quality is not 0
df.loc[df['x_quality'] != 0, ['x1', 'x2']] = np.nan

# Detect errors where x_quality is not 0
detected_errors = np.where(df['x_quality'] != 0)[0]

# Perform interpolation to fix the errors in x1
df['x1_new'] = dc.interpolation(t.astype('int64'), df['x1'], detected_errors)
df['x2_new'] = dc.interpolation(t.astype('int64'), df['x2'], detected_errors)

def z_score_filter(df, window_length, threshold ,  threshold2):
    # Υπολογισμός κυλιόμενου μέσου όρου και τυπικής απόκλισης
    m = df['x1_new'].rolling(window=window_length, min_periods=int(window_length*0.8), center=True).mean()
    s = df['x1_new'].rolling(window=window_length, min_periods=int(window_length*0.8), center=True).std()
    
    # Αποφυγή διαίρεσης με το μηδέν
    s = s.replace(0, np.nan)
    
    # Υπολογισμός του Z-score
    z_scores = (df['x1_new'] - m) / s
    
    # Εντοπισμός των άκυρων τιμών με βάση το threshold του Z-score
    invalid_data_u = z_scores > threshold
    invalid_data_d = z_scores < -threshold2
   
    # Δημιουργία νέου DataFrame όπου οι άκυρες τιμές αντικαθίστανται με NaN
    df_invalid = df.copy()
    df_invalid.loc[invalid_data_u, 'x1_new'] = np.nan
    df_invalid.loc[invalid_data_d, 'x1_new'] = np.nan
    
    return df_invalid

# Εφαρμογή
df_filtered1_invalid = z_score_filter(df, 20, 1 , 0.5)


# Αρχικά δεδομένα

plt.semilogy(df['x1_new'], color='blue')


# Φιλτραρισμένα δεδομένα

plt.semilogy(df_filtered1_invalid['x1_new'], label='Φιλτραρισμένα Δεδομένα (με Z-Score)', color='red')



