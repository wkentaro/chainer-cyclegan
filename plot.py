#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('logs/log.csv')

plt.figure(dpi=100, figsize=(16, 8))

ax = plt.subplot(231)
df.plot(x='iteration', y=['loss_G'], ax=ax)
plt.legend()

ax = plt.subplot(232)
df.plot(x='iteration', y=['loss_G_A', 'loss_G_B'], ax=ax)
plt.legend()

ax = plt.subplot(233)
df.plot(x='iteration', y=['loss_idt_A', 'loss_idt_B'], ax=ax)
plt.legend()

ax = plt.subplot(234)
df.plot(x='iteration', y=['loss_cycle_A', 'loss_cycle_B'], ax=ax)
plt.legend()

ax = plt.subplot(235)
df.plot(x='iteration', y=['loss_D_A', 'loss_D_B'], ax=ax)
plt.legend()

plt.show()
