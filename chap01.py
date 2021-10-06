#!/usr/bin/env python
# coding: utf-8

# ## ThinkDSP
# 
# This notebook contains code examples from Chapter 1: Sounds and Signals
# 
# Copyright 2015 Allen Downey
# 
# License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/)
# 

# ## Think DSP module
# 
# `thinkdsp` is a module that accompanies _Think DSP_ and provides classes and functions for working with signals.
# 
# [Documentation of the thinkdsp module is here](http://greenteapress.com/thinkdsp.html). 

# In[1]:


# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    get_ipython().system('wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py')


# ## Signals
# 
# Instantiate cosine and sine signals.

# In[2]:



import thinkdsp as td

import ryDsp

# In[ ]:





# In[3]:


from thinkdsp import CosSignal, SinSignal

cos_sig= CosSignal(freq=440, amp=1.0, offset=0)
sin_sig= SinSignal(freq=880, amp=0.5, offset=0)


# In[4]:


#
import matplotlib.pyplot as pl

x= CosSignal()
x.plot()
y= SinSignal()
y.plot()
z= x+y
z.plot()
pl.grid()
pl.xlabel('t')
pl.ylabel('x, y, z')
pl.legend(['x','y','z'])


# Plot the sine and cosine signals.  By default, `plot` plots three periods.  

# In[5]:


from thinkdsp import decorate

cos_sig.plot()
decorate(xlabel='Time (s)')


# Here's the sine signal.

# In[6]:


sin_sig.plot()
decorate(xlabel='Time (s)')


# Notice that the frequency of the sine signal is doubled, so the period is halved.
# 
# The sum of two signals is a SumSignal.

# In[7]:


mix = sin_sig + cos_sig
mix


# Here's what it looks like.

# In[8]:


mix.plot()
decorate(xlabel='Time (s)')


# ## Waves
# 
# A Signal represents a mathematical function defined for all values of time.  If you evaluate a signal at a sequence of equally-spaced times, the result is a Wave.  `framerate` is the number of samples per second.

# In[9]:


wave = mix.make_wave() #duration=0.5, start=0, framerate=11025)
wave

ryDsp.playWav(wave.ys, sr= wave.framerate)

# IPython provides an Audio widget that can play a wave.

# In[34]:


from IPython.display import Audio
audio = Audio(data=wave.ys, rate=wave.framerate)
audio


# Wave also provides `make_audio()`, which does the same thing:

# In[11]:


wave.make_audio()


# The `ys` attribute is a NumPy array that contains the values from the signal.  The interval between samples is the inverse of the framerate.

# In[12]:


print('Number of samples', len(wave.ys))
print('Timestep in ms', 1 / wave.framerate * 1000)


# Signal objects that represent periodic signals have a `period` attribute.
# 
# Wave provides `segment`, which creates a new wave.  So we can pull out a 3 period segment of this wave.

# In[13]:


period = mix.period
segment = wave.segment(start=0, duration=period*3)
period


# Wave provides `plot`

# In[14]:


segment.plot()
decorate(xlabel='Time (s)')


# `normalize` scales a wave so the range doesn't exceed -1 to 1.
# 
# `apodize` tapers the beginning and end of the wave so it doesn't click when you play it.

# In[15]:


wave.normalize()
wave.apodize()
wave.plot()
decorate(xlabel='Time (s)')


# You can write a wave to a WAV file.

# In[16]:


wave.write('temp.wav')


# `wave.write` writes the wave to a file so it can be used by an exernal player.

# In[17]:


from thinkdsp import play_wave

play_wave(filename='temp.wav', player='aplay')


# `read_wave` reads WAV files.  The WAV examples in the book are from freesound.org.  In the contributors section of the book, I list and thank the people who uploaded the sounds I use.

# In[18]:


from thinkdsp import read_wave

wave = read_wave('92002__jcveliz__violin-origional.wav')


ryDsp.playWav(wave.ys, wave.framerate)
ryDsp.plotWav(wave.ys, wave.framerate)

# In[19]:


wave.make_audio()


# I pulled out a segment of this recording where the pitch is constant.  When we plot the segment, we can't see the waveform clearly, but we can see the "envelope", which tracks the change in amplitude during the segment.

# In[20]:


start = 1.2
duration = 0.6
segment = wave.segment(start, duration)
segment.plot()
decorate(xlabel='Time (s)')


# ## Spectrums
# 
# Wave provides `make_spectrum`, which computes the spectrum of the wave.

# In[21]:


spectrum = segment.make_spectrum()


# Spectrum provides `plot`

# In[22]:


spectrum.plot()
decorate(xlabel='Frequency (Hz)')


# The frequency components above 10 kHz are small.  We can see the lower frequencies more clearly by providing an upper bound:

# In[23]:


spectrum.plot(high=10000)
decorate(xlabel='Frequency (Hz)')


# Spectrum provides `low_pass`, which applies a low pass filter; that is, it attenuates all frequency components above a cutoff frequency.

# In[24]:


spectrum.low_pass(3000)


# The result is a spectrum with fewer components.

# In[25]:


spectrum.plot(high=10000)
decorate(xlabel='Frequency (Hz)')


# We can convert the filtered spectrum back to a wave:

# In[26]:


filtered = spectrum.make_wave()


# And then normalize it to the range -1 to 1.

# In[27]:


filtered.normalize()


# Before playing it back, I'll apodize it (to avoid clicks).

# In[28]:


filtered.apodize()
filtered.plot()
decorate(xlabel='Time (s)')


# And I'll do the same with the original segment.

# In[29]:


segment.normalize()
segment.apodize()
segment.plot()
decorate(xlabel='Time (s)')


# Finally, we can listen to the original segment and the filtered version.

# In[30]:


segment.make_audio()

ryDsp.playWav(segment.ys,segment.framerate)

# In[31]:


filtered.make_audio()

ryDsp.playWav(filtered.ys, filtered.framerate)

# The original sounds more complex, with some high-frequency components that sound buzzy.
# The filtered version sounds more like a pure tone, with a more muffled quality.  The cutoff frequency I chose, 3000 Hz, is similar to the quality of a telephone line, so this example simulates the sound of a violin recording played over a telephone.

# ## Interaction
# 
# The following example shows how to use interactive IPython widgets.

# In[32]:


import matplotlib.pyplot as plt
from IPython.display import display

def filter_wave(wave, start, duration, cutoff):
    """Selects a segment from the wave and filters it.
    
    Plots the spectrum and displays an Audio widget.
    
    wave: Wave object
    start: time in s
    duration: time in s
    cutoff: frequency in Hz
    """
    segment = wave.segment(start, duration)
    spectrum = segment.make_spectrum()

    spectrum.plot(color='0.7')
    
    #spectrum.low_pass(cutoff)
    spectrum.high_pass(cutoff)
    
    
    spectrum.plot(color='#045a8d')
    decorate(xlabel='Frequency (Hz)')
    plt.show()
    
    w= spectrum.make_wave()
    
    audio = spectrum.make_wave().make_audio()
    display(audio)
    
    return w


# Adjust the sliders to control the start and duration of the segment and the cutoff frequency applied to the spectrum.

# In[33]:


from ipywidgets import interact, fixed

wave = read_wave('92002__jcveliz__violin-origional.wav')

interact(filter_wave, 
         wave= fixed(wave), 
         start=(0, 5, 0.1), 
         duration=(0, 5, 0.1), 
         cutoff=(0, 10000, 100));


# In[ ]:

wave= read_wave('92002__jcveliz__violin-origional.wav')
ryDsp.playWav(wave.ys, wave.framerate)
ryDsp.plotWav(wave.ys, wave.framerate)

#%%

    
w= filter_wave(wave, 0, 5, 1000)
ryDsp.playWav(w.ys, w.framerate)
ryDsp.plotWav(w.ys, w.framerate)

#%%
def ryFilterWav(wave, start, duration, cutoff, LowHighPass= 'lowpass'):
    """Selects a segment from the wave and filters it.
    
    Plots the spectrum and displays an Audio widget.
    
    wave: Wave object
    start: time in s
    duration: time in s
    cutoff: frequency in Hz
    """
    segment=  wave.segment(start, duration)
    spectrum= segment.make_spectrum()

    spectrum.plot(color='0.7')
    
    if LowHighPass == 'lowpass':
        spectrum.low_pass(cutoff)
    elif LowHighPass == 'highpass':
        spectrum.high_pass(cutoff)
    else:
        pass
    
    
    spectrum.plot(color='#045a8d')
    decorate(xlabel='Frequency (Hz)')
    plt.show()
    
    w= spectrum.make_wave()
    
    #audio = spectrum.make_wave().make_audio()
    #display(audio)
    
    return w

w= ryFilterWav(wave, 0, 5, 1000, LowHighPass= 'lowpass')

ryDsp.playWav(w.ys, w.framerate)
ryDsp.plotWav(w.ys, w.framerate)
