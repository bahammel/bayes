import os
import glob
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np

x_line = np.linspace(0, 10, 1000)
lines = [] 
sampled_models = [guide(None, None) for _ in range(10000)]
for m in sampled_models: 
    w, b = m.parameters() 
    lines.append(w.cpu().detach().numpy()[0]*x_line + b.cpu().detach().numpy()) 

f = plt.figure(1)
f.axes[0].set_ylim(1, 3) 
plt.figure()
for l in lines[:10]: 
    plt.plot(x_line, l) 
y_mu = np.mean(lines, axis=0)
#plt.plot(x_line, y_mu, size=10) 
plt.plot(x_line, y_mu, linestyle='dashed')
plt.autoscale() 
plt.figure() 
yy = y_data.cpu().detach().numpy().ravel()
plt.hist(yy)
five = np.percentile(yy, 5)
plt.gca().lines.pop()
plt.draw()
plt.axvline(five, color='g') 
ninefive = np.percentile(yy, 95)
plt.axvline(ninefive, color='g')
yy.std()
