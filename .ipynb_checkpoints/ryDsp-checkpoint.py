# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:17:13 2021

@author: renyu
"""

from pytube import YouTube
import scipy.io.wavfile as wf
import subprocess

import numpy as np
import matplotlib.pyplot as pl

import time


π= np.pi


defaultSampleRate= 44100
defaultTmpFile= '_tmp_.wav'

defaultUrl0= 'https://www.youtube.com/watch?v=Ptk_1Dc2iPY' # Canon
defaultUrl1= 'https://youtu.be/HBUwoAoScaQ' # 手嶌 葵《テルーの唄 》中日字幕
defaultUrl2= 'https://youtu.be/VI8zQG-yMMI' # 中島みゆき「糸」フル

defaultFile0= '_tmp_.mp3.wav'
defaultFile1= 'rySound.wav'
defaultFile2= 'L:\\中島美雪2020\\中島みゆき0102_mp3\\中島みゆき_竹の歌.mp3'
defaultFile3= 'L:\\中島美雪2020\\中島みゆき0102_mp3\\中島みゆき「この世に二人だけ」.mp3'


def getWavFromYoutube(
    url= defaultUrl1,
    play= True
    ):
    '''    
    Parameters
    ----------
    url : the url of youtube video
    play : play it or not

    Returns
    -------
    sr : sample rate
    x :  the wav data

    '''
    
    yt= YouTube(url)
    stream= yt.streams.get_audio_only()
    
    info= stream.title
    print(f'url= {url}, info= {info}')
    
    fileName= defaultTmpFile  # provide a filename for youtube downloading
    filePath= stream.download(filename= fileName)
    fileName_wav= f'{fileName}.wav'

    subprocess.run(f'ffmpeg -y -i {fileName} {fileName_wav}')    # .run() 會等
    #subprocess.Popen('ffmpeg -y -i _tmp_.mp3 _tmp_.wav') # .Popen() 不會等
    
    sr, x= wf.read(fileName_wav)

    if play is True:        
        #wf.write(fileName_wav, sr, x)
        subprocess.Popen(f'ffplay -autoexit {fileName_wav}')
    return sr, x

#sr, x= getWavFromYoutube()
#sr, x    

def playWav(x, 
            sr= defaultSampleRate, 
            _tmp_wav= defaultTmpFile):
    '''
    

    Parameters
    ----------
    x : np.ndarray , 
        DESCRIPTION. the wav data to be play
    sr : int
        sample rate
    _tmp_wav : a tmp wav file name to store x in disk
        

    Returns
    -------
    None.

    '''
    duration= len(x)/sr 
    print(f'playWav, sr= {sr}, duration= {duration:0.3f} sec')
    
    x= _ry_quantize(x)
    
    sr= int(sr)

    wf.write(_tmp_wav, sr, x)
    #subprocess.run('ffplay -autoexit _tmp_y.wav')
    subprocess.Popen(f'ffplay -autoexit {_tmp_wav}')


def playAudioFile(filename= '_tmp_.mp3'):
    '''
    play AudioFile, including .wav, .mp3, ...
    '''
    cmd= f'ffplay -autoexit "{filename}"'
    print(f'cmd= {cmd}')
    subprocess.Popen(cmd)
    return filename

import os.path

def convertAudioFile(filename= '_tmp_.mp3'):
    '''
    convert .mp3 --> .wav
    '''
    dirname= basename= os.path.dirname(filename)
    basename= os.path.basename(filename)
    outfile= f'{basename}.wav'
    cmd= f'ffmpeg -y -i "{filename}" "{outfile}"'
    print(f'cmd= {cmd}')
    
    subprocess.run(cmd)
    return outfile

def getWavFromFile_mp3(filename= '_tmp_.mp3'):
    
    #dirname=  os.path.dirname
    #basename= os.path.basename(filename)
    
    filename_wav= convertAudioFile(filename)
    sr,x= wf.read(filename_wav)
    return sr, x


#%%
# 簡單的強制型態轉換可以簡單解決，
# 但若 float 的數字範圍太小也不行，
# 需要scale至最大整數範圍之後再來強制型態轉換

def _ry_quantize(ys, 
                 bound= 2**15-1, 
                 dtype= np.int16):
    
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

    returns: quantized signal
    
    # 簡單的強制型態轉換可以簡單解決，
    # 但若 float 的數字範圍太小也不行，
    # 需要scale至最大整數範圍之後再來強制型態轉換
    """
    ys= ys/abs(ys).max()
    ys *= bound
  
    ys= ys.astype(dtype)
    
    return ys

#%%
def downSample(x, factor=2):
    y= x[::factor]
    return y

def upSample_0(x, factor=2):
    '''
    updample 比較麻煩
    '''
    
    y= np.vstack([x]*factor)*0
    for i in range(factor):
        y[i::factor]= x
    return y


def upSample(x, factor= 2):
    '''
    use linear interpolation to do the upsampling 

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.the input wav
    factor : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    y : TYPE
        DESCRIPTION. the output upsampled wav 

    '''
    
    y= np.vstack([x]*factor)*0

    dx= (x[1::1]-x[0:-1:1])
    for i in range(factor):
        y[i:-factor:factor]= x[0:-1:1] + dx*i/factor
    y[-factor:]= x[-1]
    
    return y

#%%
def plotWav(x, sr= 1, **kwargs):

    ts= np.arange(len(x))/sr
    pl.plot(ts, x,**kwargs)
    pl.grid()
    #pl.legend(['0','1'])
    if sr==1:
        pl.xlabel('sample')
    else:
        pl.xlabel('sec')
#%%

def echoWav(x, 
            sr= 44100, 
            rightLeftShift= 0.05):
    
    rightLeftShift= int(rightLeftShift*sr)
    
    z= x*0
    z[:,     0]= x[:,      0] 
    z[rightLeftShift:, 1]= x[:-rightLeftShift, 1]
    return z

#%%
def zeroPaddingWav(x, nFFT= 1024):
    '''
    在 x 後面
    補0至 nFFT 的整數倍
    '''
    #
    # zero_padding the last frame to be lenght = nFFT
    #
    redundancy= len(x)%nFFT
    if redundancy!=0:
        x00= [x, 
              [x[0]*0]
              *(nFFT-redundancy)
             ]
        #x= np.vstack(x00)
        x= np.concatenate(x00, axis=0)        
    return x

#%%
import numpy.fft

def shortTimeFFT(x, nFFT= 1024):
    
    assert x.ndim == 1
    
    x= zeroPaddingWav(x, nFFT)
    z= x.reshape(-1, nFFT)
    Z=  numpy.fft.fft(z)
    
    return Z

def shortTimeFFT_inv(Z):
    
    z= numpy.fft.ifft(Z)
    x= z.real.flatten()
    
    return x
    
def specWav(x, nFFT= 1024):
    
    if x.ndim==2:
        x= x.mean(axis=1)
        x= x.flatten()
    
    X= shortTimeFFT(x, nFFT)
    X= np.abs(X)
    X= np.log(X)
    
    X= X[:, -nFFT//2:]
    X= X.T
    
    pl.imshow(X)
    
    return X
    


def filterWav(x, 
              sr= 44100, 
              cutoff_freq= 1000, 
              lowpass=True):
    '''
    cutoff_ratio: the ratio of sampling frequency to cut off
    the max is 1/2
    '''
    
    assert x.ndim == 1 # currently only deal with mono(ch=1) audio
    
        
    assert cutoff_freq/sr < 1/2
    assert cutoff_freq/sr > 0
        
    X= numpy.fft.fft(x)
    
    N= X.shape[0]
    cutoff= int(N*cutoff_freq/sr)

    if lowpass==True:
        print(f'lowpass, cutoff_freq= {cutoff_freq} Hz')
        X[cutoff:-cutoff]= 0
    else:
        print(f'highpass, cutoff_freq= {cutoff_freq} Hz')
        X[:cutoff]=  0
        X[-cutoff:]= 0

    xx= numpy.fft.ifft(X)
    xx= xx.real
    return xx
#%%

def genSinSignal(T=1, f= 440, A=1, ϕ=0, sr= 44100):
    ts= np.linspace(0, T, sr*T)
    θ= 2 *π  *f *ts + ϕ
    ys= A * np.sin(θ)
    return sr, ys

def genCosSignal(T=1, f= 440, A=1, ϕ=0, sr= 44100):
    ts= np.linspace(0, T, sr*T)
    θ= 2 *π  *f *ts + ϕ
    ys= A * np.cos(θ)
    return sr, ys

def genSquareSignal(T=1, f= 440, A=1, ϕ=0, sr= 44100):
    sr, ys= genSinSignal(T, f, A, ϕ, sr)
    ys= np.sign(ys)
    return sr, ys

def genTriangleSignal(T=1, f= 440, A=1, ϕ=0, sr= 44100):
    ts= np.linspace(0, T, sr*T)
    θ= 2 *π  *f *ts + ϕ
    cycles= θ/(2 *π)
    frac, _ = np.modf(cycles)
    ys= np.abs(frac-.5) - .5/2
    ys= ys*4*A
    
    #ys= A * np.sin(θ)
    
    return sr, ys

def genSawtoothSignal(T=1, f= 440, A=1, ϕ=0, sr= 44100):
    ts= np.linspace(0, T, sr*T)
    θ= 2 *π  *f *ts + ϕ
    cycles= θ/(2 *π)
    frac, _ = np.modf(cycles)
    ys= np.abs(frac) - .5
    ys= ys*2*A
    
    #ys= A * np.sin(θ)
    
    return sr, ys


#%%
#%%
if __name__=='__main__':
    
    sr, wav= getWavFromYoutube()
    
    #time.sleep(1)
    input('after playing, press any key to continue..')
    
    x= wav[sr*100:sr*120, :]
    plotWav(wav, sr*10//9)
    playWav(wav, sr*10//9)
    
    #time.sleep(1)
    input('after playing, press any key to continue..')
    
    x= wav[sr*100:sr*120, 0]
    plotWav(wav, sr*9//10)
    playWav(wav, sr*9//10)
    
    #time.sleep(1)
    input('after playing, press any key to continue..')
    
    x=  wav[sr*100:sr*120, 0]
    xx= filterWav(x, sr, cutoff_freq= 1000, lowpass=True)
    playWav(xx,sr)
    plotWav(xx,sr)
    
    #time.sleep(1)
    input('after playing, press any key to continue..')

    x=  wav[sr*100:sr*120, 0]
    xx= filterWav(x, sr, cutoff_freq= 1000, lowpass=False)
    playWav(xx,sr)
    plotWav(xx,sr)
    
    #time.sleep(1)
    input('after playing, press any key to continue..')
    
    
#%%
#%%
#%%

#%%

class ryWav:
    def __init__(self, x= None, sr= None):
                
        self.x= x
        self.sr= sr
        
    def plot(self):
        plotWav(self.x)

    def play(self):
        playWav(self.x, self.sr)
    
    def getFromYoutube(self, url= defaultUrl1):
        sr, x= getWavFromYoutube(url)
        self.sr= sr
        self.x=  x
        return sr, x
    
    def getFromFile(self, file= defaultFile1):
        sr, x= wf.read(file)
        self.sr= sr
        self.x=  x
        return sr, x
    
    def getFromFile_mp3(self, file= defaultFile2):
        
        #sr, x= wf.read(file)
        sr, x= getWavFromFile_mp3(file) 
        
        self.sr= sr
        self.x=  x
        return sr, x
    
    def segment(self, t0=0, t1=None):
        
        if self.x.ndim==2:
            x= self.x[t0:t1,:]
        
        elif self.x.ndim==1:
            x= self.x[t0:t1]
        
        else:
            return None
        
        y= ryWav(x= x, sr= self.sr)
        return y
    
    def spectrogram(self, nFFT= 1024):
        
        X= specWav(self.x, nFFT)
        
        self.X= X
        return X
        

#%%        