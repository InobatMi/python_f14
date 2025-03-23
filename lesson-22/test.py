import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dev_x = [21,32,23,24,35,26,27]
dev_y = [100,300,450,500,750,800,900]


plt.plot(dev_x, dev_y, marker='.', linestyle='-.', color='k', linewidth=1)
plt.title('Salary by age')

plt.show()