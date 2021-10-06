#!/usr/bin/env python
# coding: utf-8

# ## ThinkDSP
# 
# This notebook contains code examples from Chapter 2: Harmonics
# 
# Copyright 2015 Allen Downey
# 
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)

# In[1]:


# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    get_ipython().system('wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py')


# ## Waveforms and harmonics
# 
# Create a triangle signal and plot a 3 period segment.

# In[2]:


from thinkdsp import TriangleSignal
from thinkdsp import decorate

signal=   TriangleSignal(200)
duration= signal.period*3
segment=  signal.make_wave(duration, framerate=10000)

#%%
segment.plot()
decorate(xlabel='Time (s)')


# Make a wave and play it.

# In[3]:

wave= signal.make_wave(duration=0.5, framerate=10000)

wave.apodize()
wave.make_audio()

#%%

signal=   TriangleSignal()
wave=     signal.make_wave()#duration=5, framerate= 44100)

import ryDsp
ryDsp.playWav(wave.ys)

#%%
# Compute its spectrum and plot it.

# In[4]:


spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')


# Make a square signal and plot a 3 period segment.

# In[5]:


from thinkdsp import SquareSignal

signal = SquareSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')


# Make a wave and play it.

# In[6]:


wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()


# Compute its spectrum and plot it.

# In[7]:


spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')


# Create a sawtooth signal and plot a 3 period segment.

# In[8]:


from thinkdsp import SawtoothSignal

signal = SawtoothSignal(200)
duration = signal.period*3
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')


# Make a wave and play it.

# In[9]:


wave = signal.make_wave(duration=0.5, framerate=10000)
wave.apodize()
wave.make_audio()


# Compute its spectrum and plot it.

# In[10]:


spectrum = wave.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')


# ### Aliasing
# 
# Make a cosine signal at 4500 Hz, make a wave at framerate 10 kHz, and plot 5 periods.

# In[11]:


from thinkdsp import CosSignal

signal = CosSignal(4500)
duration = signal.period*5
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')


# Make a cosine signal at 5500 Hz, make a wave at framerate 10 kHz, and plot the same duration.
# 
# With framerate 10 kHz, the folding frequency is 5 kHz, so a 4500 Hz signal and a 5500 Hz signal look exactly the same.

# In[12]:


signal = CosSignal(5500)
segment = signal.make_wave(duration, framerate=10000)
segment.plot()
decorate(xlabel='Time (s)')


# Make a triangle signal and plot the spectrum.  See how the harmonics get folded.

# In[13]:


signal = TriangleSignal(1100)
segment = signal.make_wave(duration=0.5, framerate=10000)
spectrum = segment.make_spectrum()
spectrum.plot()
decorate(xlabel='Frequency (Hz)')


# ## Amplitude and phase
# 
# Make a sawtooth wave.

# In[14]:


signal = SawtoothSignal(500)
wave = signal.make_wave(duration=1, framerate=10000)
segment = wave.segment(duration=0.005)
segment.plot()
decorate(xlabel='Time (s)')


# Play it.

# In[15]:


wave.make_audio()


# Extract the wave array and compute the real FFT (which is just an FFT optimized for real inputs).

# In[16]:


import numpy as np

hs = np.fft.rfft(wave.ys)
hs


# Compute the frequencies that match up with the elements of the FFT.

# In[17]:


n = len(wave.ys)                 # number of samples
d = 1 / wave.framerate           # time between samples
fs = np.fft.rfftfreq(n, d)
fs


# Plot the magnitudes vs the frequencies.

# In[18]:


import matplotlib.pyplot as plt

magnitude = np.absolute(hs)
plt.plot(fs, magnitude)
decorate(xlabel='Frequency (Hz)')


# Plot the phases vs the frequencies.

# In[19]:


angle = np.angle(hs)
plt.plot(fs, angle)
decorate(xlabel='Phase (radian)')


# ## What does phase sound like?
# 
# Shuffle the phases.

# In[20]:


import random
random.shuffle(angle)
plt.plot(fs, angle)
decorate(xlabel='Phase (radian)')


# Put the shuffled phases back into the spectrum.  Each element in `hs` is a complex number with magitude $A$ and phase $\phi$, with which we can compute $A e^{i \phi}$

# In[21]:


i = complex(0, 1)
spectrum = wave.make_spectrum()
spectrum.hs = magnitude * np.exp(i * angle)


# Convert the spectrum back to a wave (which uses irfft).

# In[22]:


wave2 = spectrum.make_wave()
wave2.normalize()
segment = wave2.segment(duration=0.005)
segment.plot()
decorate(xlabel='Time (s)')


# Play the wave with the shuffled phases.

# In[23]:


wave2.make_audio()


# For comparison, here's the original wave again.

# In[24]:


wave.make_audio()


# Although the two signals have different waveforms, they have the same frequency components with the same amplitudes.  They differ only in phase.

# ## Aliasing interaction
# 
# The following interaction explores the effect of aliasing on the harmonics of a sawtooth signal.

# In[25]:


def view_harmonics(freq, framerate):
    """Plot the spectrum of a sawtooth signal.
    
    freq: frequency in Hz
    framerate: in frames/second
    """
    signal = SawtoothSignal(freq)
    wave = signal.make_wave(duration=0.5, framerate=framerate)
    spectrum = wave.make_spectrum()
    spectrum.plot(color='C0')
    decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
    display(wave.make_audio())


# In[26]:


from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

slider1 = widgets.FloatSlider(min=100, max=10000, value=100, step=100)
slider2 = widgets.FloatSlider(min=5000, max=40000, value=10000, step=1000)
interact(view_harmonics, freq=slider1, framerate=slider2);


# In[ ]:





# In[ ]:




