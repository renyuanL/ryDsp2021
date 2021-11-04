# -*- coding: utf-8 -*-
"""
ryMusic.py

Created on Fri Nov  5 02:36:39 2021

@author: renyu
"""

#
# ryMusic.py
#



from ryDsp   import *
from ryMusic_melody import *


def musicNoteFreq(ts,  keyShift= 0, melodyId=0):       
    '''
    if melodyId==0:
        melody= melody0
    elif melodyId==1:
        melody= melody1
    elif melodyId==2:
        melody= melody2
    elif melodyId==3:
        melody= melody3
    else:
        melody= melody0
    '''
    
    melody= eval(f'melody{melodyId}')


    melody= melody.split()
    melody= [noteName[k] for k in melody]

    nNotes= len(melody)

    fs= np.zeros_like(ts)

    dur= len(fs)

    #nNotes= 32
    dt=  dur//nNotes
    kL= []
    for n in range(nNotes):
        #k= np.random.choice(notes)
        k= melody[n] + keyShift
        kL += [k]

        fs[dt*n:
           dt*(n+1)
          ]= 440*2**(k/12)

    print(f'the notes to play, kL= {kL}')

    return fs

#ts= np.linspace(0,1,1000)
#musicNoteFreq(ts, melodyId=5).shape


def genMusicSignal(
    T=  10,    # sec 
    nNotes= 32,
    
    keyShift= 0,
    melodyId= 1,
    
    A=  1,    # amplitude
    sr= 44100 # Hz, sampling rate, samples/sec
    ):
    
    # specify the time-series for the given duration
    ts= np.linspace(0, T, sr*T)
    
    #
    # specify the freq as fs= f(ts), 
    #
    # f() can be any function, 
    # boundary conditions 
    # f(t0)==f0
    # f(t1)==f1
    #
    

    fs= musicNoteFreq(ts, keyShift, melodyId)

    
    # radian frequency
    ws= 2*π*fs
    
    #
    # θ = Integrate (w(t) dt)
    #
    dt=  T/len(ts)
    θ=   np.cumsum(ws)*dt  # this is mimic the integration 
    
    #
    # finally, generate the signal
    #
    ys= A * np.sin(θ)
    
    return sr, ys

if __name__=='__main__':
    
    sr, x1= genMusicSignal(T= 20, keyShift=   0+5,  melodyId=1, sr= 8000)
    sr, x2= genMusicSignal(T= 20, keyShift=   0+5,  melodyId=2, sr= 8000)
    sr, x3= genMusicSignal(T= 20, keyShift=   0+5,  melodyId=3, sr= 8000)
    sr, x4= genMusicSignal(T= 20, keyShift= -24+5,  melodyId=4, sr= 8000)
    sr, x5= genMusicSignal(T= 20, keyShift= -12+5,  melodyId=5, sr= 8000)
    
    x= np.stack([x2, x3, x4, x5], axis=1)
    
    
    playWav(x1, sr, _tmp_wav= 'x1')
    
    
    #playWav(x2, sr)
    #playWav(x3, sr)
    #playWav(x4, sr)
    #playWav(x5, sr)
    playWav(x, sr, _tmp_wav= 'x')
#%%    
    pl.figure()    
    rySpectrogram(x1)
    
    pl.figure()
    rySpectrogram(x2)
    
    pl.figure()
    rySpectrogram(x3)
    
    pl.figure()
    rySpectrogram(x4)
    
    pl.figure()
    rySpectrogram(x5)    
