#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/10/2025

Lachlan Perris 

"""

import numpy as np
import numbers
import scipy.interpolate
import scipy
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from datetime import tzinfo, datetime
from copy import deepcopy
#import MLfunc
        
# Calculate wave spectrum

# tvec, bursttvec, tstart, tstop in UNIX epoch time

def calc_ws(tvec,pvec,fs,mode,bursttvec=None,atm=None,tstart=None,tstop=None,zmab=None,nfft=None,tint=None):


    rho = 1025
    g   = 9.81    

    ws  = {}
    
    # Default values
    if tstart is None:
        tstart = datetime.utcfromtimestamp(tvec[0]).replace(microsecond=0, second=0, minute=0).timestamp() # Round to hour
    if tstop is None:
        tstop  = tvec[-1]
    if zmab is None:
        zmab = 0
    if tint is None:
        tint = 30*60# 30 minutes default
    if nfft is None:
        if fs == 1:
            nfft = 2**9
        elif fs == 2:
            nfft = 2**10

    # Do wave analysis over bursts or chosen time interval tint
    if mode == 'wave': # Use burst tvec
        mtime = bursttvec.copy() 
    elif mode == 'continuous': # Get interpolated tvec 
        mtime = np.arange(tstart,tstop, tint)
        # If no tstop was used, remove the last time step as it will most likely be incomplete
        if tstop == tvec[-1]:
            mtime = mtime[0:-2] 

    # Correct for atmospheric pressure
    if atm is None:                              # If no atmoshperic value is given, use 10.13
        atm = 10.13
        pres = pvec.copy() - atm
    elif isinstance(atm,numbers.Number) == True: # If atmospheric scalar is given, use this
        atm = atm
        pres = pvec.copy() - atm
    elif isinstance(atm, dict) and ('tvec' in atm.keys()) and ('pres' in atm.keys()):
        pres = pvec.copy() - np.interp(tvec, atm['tvec'], atm['pres'])
        #pres[pres<0] = 0
    
    pres[pres<0] = np.nan# Is this a good idea?

    # Pre-allocate arrays
    f       = fs*np.array(range(0,int(nfft/2)+1))/nfft; # Frequencies
    df      = f[1]-f[0]
    
    h       = np.empty(len(mtime)-1)
    Pxx     = np.empty((len(mtime)-1,len(f)))
    PUxx    = np.empty((len(mtime)-1,len(f)))
    Fxx     = np.empty((len(mtime)-1,len(f)))
    As_ss   = np.empty(len(mtime)-1)
    Sk_ss   = np.empty(len(mtime)-1)
    As_lf   = np.empty(len(mtime)-1)
    Sk_lf   = np.empty(len(mtime)-1)
    
    h[:]       = np.nan
    Pxx[:]     = np.nan
    PUxx[:]    = np.nan
    Fxx[:]     = np.nan
    As_ss[:]      = np.nan
    Sk_ss[:]      = np.nan
    As_lf[:]      = np.nan
    Sk_lf[:]      = np.nan


    # Get frequencies
    f_ww = [0.2, fs/2] # Up to Nyquist frequency
    f_ss = [0.04,0.2]
    f_ig = [0.004,0.04] 
    f_vlf= [0.001,0.004] # Check these frequencies
    
    # Wave analysis
    # Run to second-to-last time step, as last time interval is most likely to be shortened
    

    for i in range(len(mtime)-1):
        
        # Find correct time interval
        if i == 0:
            idstart = find_nearest_fast(tvec, mtime[i])
            idstop  = find_nearest_fast(tvec, mtime[i+1])
        elif i > 0: # Speed it up a little bit, only need to search for one new index
            idstart = deepcopy(idstop)
            idstop  = find_nearest_fast(tvec, mtime[i+1])

        
        if idstart == idstop: # In case time period is outside of instrument tvec
            continue 

        # Pre-allocate array with pressure measurements. Do this in case of doing burst measurements on mtime vector e.g., 64 minutes burst on vector every 30 minutes. Two first 30-minute intervals should be calculated, third one only has 4 minutes of data, rest should be nans.
        #if (bursttvec[1]-bursttvec[0]).total_seconds()/60 == float(tint.split('min')[0]) something like this?
        nump        = int((mtime[i+1]-mtime[i])*fs) # Problem if on actual bursttvec - need to fix! Right now allocates too many entries to this vector 
        tmp_pres    = np.empty(nump)
        tmp_pres[:] = np.nan
        
        # Find time series, mean depth
        tmp_pres[0:pres[idstart:idstop].shape[0]] = pres[idstart:idstop].copy()
        
        if any(np.isnan(tmp_pres)): # In case of nans (i.e., errors or not full time series during interval), skip time steps 
            continue

        
        # Pwelch
        h[i]     = np.mean(tmp_pres)+zmab
        ftmp, Pxx[i,:] = signal.welch(tmp_pres, fs=fs, nperseg=nfft, return_onesided=True, detrend='linear')

        if (f == ftmp).all() is False:
            breakpoint()
            
        # Pressure correction
        k                = get_wavenumber(2*np.pi*f, h[i])
        k[0]             = 0
        prcorr           = 1E4*np.cosh(k*h[i])/(rho*g*np.cosh(k*zmab))
        prcorr[prcorr>5] = 1
        
        Pxx[i,:] = Pxx[i,:].copy()*(prcorr**2) # Normalize variance?
        
        # Calculate wave velocities
        PUxx[i,:] = 4 * np.pi**2 * f**2 * Pxx[i,:] * np.cosh(0)**2 / np.sinh(k*h[i])**2
        
        # Calculate wave flux (no directionality)
        c        = np.sqrt(g/k*np.tanh(k*h[i]))
        n        = 0.5*(1+2*k*h[i]/np.sinh(2*k*h[i]))
        cg       = n*c
        Fxx[i,:] = Pxx[i,:] * cg * rho * g
        
        # Get asymmetry, skewness
        # First, get surface elevation from pressure
        eta         = prCorr(tmp_pres, fs, zpt=zmab)
        eta         = signal.detrend(eta,type='linear')

        eta_ss = butter_filter(eta, np.array([f_ss[0], f_ss[1]]), fs=fs, btype='bandpass', order=5)
        eta_lf = butter_filter(eta, np.array([f_ig[0], f_ig[1]]), fs=fs, btype='bandpass', order=5)
        
        #Remove 5 mins before/after because of edge effects of butter filter
        cutmins = 5
        cutind  = int(cutmins*60//fs)
        eta_ss = eta_ss[cutind:-cutind]
        eta_lf = eta_lf[cutind:-cutind]

        Sk_ss[i]    = np.mean(eta_ss**3) / np.mean(eta_ss**2)**(3/2)
        As_ss[i]    = np.mean(np.imag(signal.hilbert(eta_ss))**3) / np.mean(eta_ss**2)**(3/2)
        Sk_lf[i]    = np.mean(eta_lf**3) / np.mean(eta_lf**2)**(3/2)
        As_lf[i]    = np.mean(np.imag(signal.hilbert(eta_lf))**3) / np.mean(eta_lf**2)**(3/2)

    # Save wave spectrum
    ws['tvec']  = mtime[0:-1] #mtime[0:-1]+(mtime[1]-mtime[0])/2
    ws['depth'] = h
    ws['f']     = f
    ws['Pxx']   = Pxx
    ws['PUxx']  = PUxx
    ws['Fxx']   = Fxx

    ##  --- Wave energies --- ##
    # Find energies per frequency
    ind = np.argwhere(np.logical_and(ws['f']>=f_ww[0], ws['f']<=f_ww[1])).squeeze()
    ws['Hs_ww']   = 4 * np.sqrt(np.sum(ws['Pxx'][:,ind],axis=1)*df)            
    
    ind = np.argwhere(np.logical_and(ws['f']>=f_ss[0], ws['f']<f_ss[1])).squeeze()
    ws['Hs_ss']   = 4 * np.sqrt(np.sum(ws['Pxx'][:,ind],axis=1)*df)
    
    ind = np.argwhere(np.logical_and(ws['f']>=f_ig[0], ws['f']<f_ig[1])).squeeze()
    ws['Hs_ig']   = 4 * np.sqrt(np.sum(ws['Pxx'][:,ind],axis=1)*df)
     
    if any(f[1::]<f_vlf[1]): # Averaging period/bursts may not be long enough to capture below 0.004 Hz (250s), only include vlf if sampling/averaging period is long enough 
        ind = np.argwhere(np.logical_and(ws['f']>=f_vlf[0], ws['f']<f_vlf[1])).squeeze()
        if ind.shape == (): # If only one frequency is within vlf band, get an error when trying to sum (since there is only one entry to sum over). Then nan, not enough data points
            ws['Hs_vlf']    = np.empty((ws['Hs_ig'].shape[0],ws['Hs_ig'].shape[1]))
            ws['Hs_vlf'][:] = np.nan #4 * np.sqrt(ws['Pxx'][:,ind]*df)            
        else:
            ws['Hs_vlf']  = 4 * np.sqrt(np.sum(ws['Pxx'][:,ind],axis=1)*df)            
    
    # All wave energy
    ws['Hs_all']  = 4 * np.sqrt(np.sum(ws['Pxx'],axis=1)*df)
    
    ##  --- Wave periods --- ##
    m0 = np.sum(ws['Pxx'],axis=1)*df
    m1 = np.sum(f*ws['Pxx'],axis=1)*df
    ws['Tm'] = m0/m1
    ws['Tp'] = 1/f[np.argmax(Pxx,axis=1)]

    ##  --- Other wave quantities --- ##
    # Wave orbital velocity at the bottom
    ws['urms'] = np.sqrt( np.nansum( PUxx,axis=1 )*df )
    
    # Wave energy flux
    ws['Fw']   = np.nansum( Fxx,axis=1 ) * df
    
    # Skewness and asymmetry - NOT SURE THESE ARE CORRECTLY CALCULATED
    ws['As_ss'] = As_ss
    ws['Sk_ss'] = Sk_ss
    ws['As_lf'] = As_lf
    ws['Sk_lf'] = Sk_lf

    return ws

def calc_hs_from_ws(ws, fmin=None, fmax=None):
    f = np.asarray(ws['f'])
    Pxx = np.asarray(ws['Pxx'])
    if fmin is None: fmin = float(f.min())
    if fmax is None: fmax = float(f.max())
    idstart = find_nearest_fast(f, fmin)
    idstop  = find_nearest_fast(f, fmax)
    if idstart > idstop: idstart, idstop = idstop, idstart
    df = f[1] - f[0]
    hs_int = np.sum(Pxx[:, idstart:idstop+1], axis=1) * df
    ws['Hs_custom'] = 4 * np.sqrt(hs_int)
    return ws
    

#%% Auxillary functions 

"""
From Falk Feddersen get_wavenumber.m
"""

def get_wavenumber(omega,dep):
    # From Falk Feddersen get_wavenumber.m

    g = 9.81
    
    k = omega/np.sqrt(g*dep)

    f = g*k*np.tanh(k*dep) - omega**2

    if np.prod(f.shape) == 1:
        while abs(f)>10E-10:
            dfdk = g*k*dep*(1/np.cosh(k*dep))**2 + g*np.tanh(k*dep)
            k    = k - f/dfdk
            f    = np.asarray(g*k*np.tanh(k*dep) - omega**2)
    else:
        while max(abs(f))>10E-10:
            dfdk = g*k*dep*(1/np.cosh(k*dep))**2 + g*np.tanh(k*dep)
            k    = k - f/dfdk
            f    = np.asarray(g*k*np.tanh(k*dep) - omega**2)

    return k


def moving_average(a, n=3):
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n


##############################################################################

def get_wavenumber_poly(omega,dep):
    
    g = 9.81
    
    f = omega/2/np.pi
    
    dum1 = (omega**2)*dep/g
    dum2 = dum1 + (1.0 + 0.6522 * dum1 + 0.4622 * (dum1**2) + 0.0864 * (dum1**4) + 0.0675 * (dum1**5))**(-1)
    dum3 = np.sqrt(9.81 * dep * dum2**(-1))/f
    
    return 2 * np.pi * dum3**(-1)

"""
From pr_corr.m
"""    
def prCorr(p,fs,zpt=0,maxattenuationfactor=5):

    h     = np.mean(p)
    pvar  = scipy.signal.detrend(p.squeeze(),type='linear')
    
    P = fft(pvar)
    f = fftfreq(len(pvar),1/fs)
    
    k = get_wavenumber(2*np.pi*f,h)
    
    Kpt = np.cosh(k*zpt)/np.cosh(k*h)
    
    Kpt[1/Kpt>maxattenuationfactor] = 1
    
    Pcor    = P/Kpt
    Pcor[0] = 1
    eta  = np.real(ifft(Pcor)) + h
    
    return eta

"""
  
"""     
def arrayminmax(*args):

    mn = []
    mx = []
    for i in range(len(args)):
        mn.append(np.min(args[i]))
        mx.append(max(args[i]))
        
    mn = np.min(mn)
    mx = np.max(mx)
    
    return mn, mx

# Just returns index of nearest  
def find_nearest_fast(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx 

# Returns index and value of nearest
def find_nearest(items, pivot):
    return min(enumerate(items), key=(lambda x: abs(x[1] - pivot)) )


# From https://stackoverflow.com/questions/39032325/python-high-pass-filter

def butter_filt(cutoff, fs, btype, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype)
    return b, a


def butter_filter(data, cutoff, fs, btype, order=5):
    b, a = butter_filt(cutoff, fs, btype, order=order)
    y = signal.filtfilt(b, a, data)
    return y



def nanrms(x):
    
    return np.sqrt(np.mean((x-np.nanmean(x))**2))






