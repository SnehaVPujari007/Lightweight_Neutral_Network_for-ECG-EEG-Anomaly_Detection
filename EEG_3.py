import torch
import arff2pandas as a2p
from pandas import DataFrame;
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
def plot_time_series_class(data,class_names,ax,n_steps=10):
    
    print(class_names)
    time_series_df=pd.DataFrame(data)
    smooth_path=time_series_df.rolling(n_steps).mean()
    path_deviation = 2*time_series_df.rolling(n_steps).std()
    under_line=(smooth_path-path_deviation)[0]
    over_line=(smooth_path+path_deviation)[0]
    ax.plot(smooth_path,linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    ax.set_title(class_names)
    
    
with open('EEG.arff') as f:
  train = a2p.load(f)
  print(train.head())
with open('EEG.arff') as f:
  test = a2p.load(f)
  print(test.head())
  
data, meta = arff.loadarff('EEG_TEST.arff') 
df = DataFrame(data,columns = meta.names())
df = train._append(test)
df = df.sample(frac=1.0)
print(df.shape)
CLASS_NORMAL = 1
class_names = ['NORMAL','RTTD','']
new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns
print(df.target.value_counts())
x= ['NORMAL','RTTD','']
#y = [df.loc[df.target == '1','target'].count(),df.loc[df.target == '2','target'].count(),df.loc[df.target == '3','target'].count()]
y=[3400.8,2000.56,0.0]
print(y)
sns.set()
sns.barplot(x=x,y=y)
plt.xlabel('Target')
plt.ylabel('Counts')
plt.show()


classes=df.target.unique()
fig,axs=plt.subplots(
nrows=len(classes)//3+1,
ncols=3,
sharey=True,
figsize=(10,4)
)


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

for i, cls in enumerate(classes):
    ax = axs[i]
    data = df[df.target == cls]\
        .drop(labels='target', axis=1)\
        .mean(axis=0)\
        .to_numpy()
    data[i]+=2000
   
    plot_time_series_class(data, class_names[i], ax)  # Pass the correct class name and axis
   

if len(classes) < len(axs):
    for j in range(len(classes), len(axs)):
        fig.delaxes(axs[j])
fig.tight_layout()
plt.show()

