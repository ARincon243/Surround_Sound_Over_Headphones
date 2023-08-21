import sofa                       # Read Sofa HRTFs
import librosa                    # Resample Function
import sys, glob
import numpy as np
import soundfile as sf
import scipy.signal as signal
from collections import deque

'''Read Audio and Converted'''
class Audio:
    def __init__(self, Input, fs):
        self.Input = Input
        self.fs = fs
        self.xtmp, self.ytmp, self.mono = self.read()
        
    def read(self):
        [data, fs_s] = sf.read(self.Input)
        if fs_s != self.fs:
            data = librosa.core.resample(y = data.T, orig_sr = fs_s, target_sr = self.fs).T
        if len(data.shape)==1:
            N = np.shape(data)[0]
            Nchannels = 1
            print('1 channel')
            xtmp = data/2**15
            return xtmp
        else:
            # number of channels
            N, Nchannels = data.shape
            if Nchannels==2:
                print('2 channels')
                xtmp = data[:,0]/2**15      # left
                ytmp = data[:,1]/2**15      # right
                mono = librosa.to_mono(y = data.T)/2**15
                return xtmp, ytmp, mono
            
'''Crossover filters design'''
class CrossoverFilter:
    def __init__(self, f_name, f_order, f_type, fc, inv, fs):
        if type(inv) is not bool:
            raise ValueError(
                'self.inverted polarity {} is incorrect'.format(inv))
        if f_type not in ("low", "high"):
            raise ValueError(
                'The filter type {} is incorrect'.format(f_type))
        if f_name not in ("Butter", "LR"):
            raise ValueError(
                'The filter name {} is incorrect'.format(f_name))
        if f_name == "LR" and (f_order%2) != 0:
            raise ValueError(
                'The order {} is incorrect for a Linkwitz–Riley Filter'.format(f_order))
        self.f_name = f_name 
        self.f_order = f_order 
        self.f_type = f_type 
        self.inv = inv
        self.fc = fc
        self.fs = fs
        
        self.b, self.a = self.set_filter_coefficients()
    
    def set_filter_coefficients(self):
        Wn = self.fc/(self.fs/2)
        if self.f_name == "Butter":
            b, a = signal.butter(self.f_order, Wn, btype=self.f_type)
            if self.inv == True:
                b = -b
        elif self.f_name == "LR":                          
            b, a = signal.butter(int(self.f_order/2), Wn, btype=self.f_type)
            b = np.convolve(b, b)
            a = np.convolve(a, a)  
            if self.inv == True:
                b = -b
        return b, a

'''Loudspeakers simulation'''
class Loudspeaker:
    def __init__(self, Re, Le, Bl, fr, Mms, Kms, Qms, Sd, fs):
        self.Re = Re 
        self.Le = Le 
        self.Bl = Bl 
        self.Qms = Qms
        self.fr = fr
        self.Mms = Mms 
        self.Kms = 1/Kms
        self.Sd = Sd
        self.fs = fs
        # calculate the digital filter coefficients
        self.b, self.a = self.digital_filter_coefficients()
    
    
    def analog_filter_coefficients(self): 
        Rms = 2*np.pi*self.fr*self.Mms/self.Qms
        B = [self.Bl, 0, 0]
        A = np.zeros(4)
        A[0] = self.Le*self.Mms
        A[1] = self.Re*self.Mms + self.Le*Rms
        A[2] = self.Re*Rms + self.Le*self.Kms + self.Bl**2 
        A[3] = self.Re*self.Kms
        return B, A
        
    def digital_filter_coefficients(self):
        B, A = self.analog_filter_coefficients()
        b, a = signal.bilinear(B, A, fs=self.fs) 
        return b, a
    
''' HRTF '''    
class oneHRTF:
    def __init__(self, HRTF_Choose, target_fs):
        self.HRTF_Choose = HRTF_Choose
        self.fs = target_fs
        
        self.fs_h, self.positions, self.data_H = self.read_HRTF()
    
    def read_HRTF(self):
        data_H = sofa.Database.open(self.HRTF_Choose)
        fs_H = data_H.Data.SamplingRate.get_values()[0]
        positions = data_H.Source.Position.get_values(system='spherical')
    
        return fs_H, positions, data_H
    
    def one_position_configuration(self, azimuth, elevation, in_signal):
    
        if (len(in_signal) != 2):
            in_signal = [in_signal, in_signal]
        
        hrtf_idx = np.array(np.where((self.positions[:,0] == azimuth) & (self.positions[:,1] == elevation)),dtype = int)
        
        # Retrieve HRTF data for angle
        H = np.zeros([self.data_H.Dimensions.N, 2])        # 3D Binaural Mix 
        H[:,0] = self.data_H.Data.IR.get_values(indices = {"M":int(hrtf_idx), 'R':0, 'E':0})
        H[:,1] = self.data_H.Data.IR.get_values(indices = {"M":int(hrtf_idx), 'R':1, 'E':0})
        
        # Resample with our target Fs
        if self.fs_h != self.fs:
            H = librosa.core.resample(y = H.T, orig_sr = self.fs_h, target_sr = self.fs).T
        
        rend_L = signal.fftconvolve(in_signal[0], H[:,0], mode='same')
        rend_R = signal.fftconvolve(in_signal[1], H[:,1], mode='same')
        
        return rend_L, rend_R
        
class _Reverb_:
    def __init__(self, target_fs):
        self.target_fs = target_fs
        
        self.delay, self.vectorCircular, self.N = self.parameters()
        
        self.pi, self.gi = self.IIRFilter()
        
    def parameters(self):
        delay = [733, 1129, 2311, 3469, 3797, 4987, 5347, 7307]
        vectorCircular = [-0.2338, -0.1289, -0.3208, -0.7270, 0.5052, 0.0488, 0.0493, -0.1928]
        N = len(delay)
        
        return delay, vectorCircular, N
        
    def IIRFilter(self):
        
        T_60_DC=1.25
        T_60_Ny=0.7
        pi = np.zeros(self.N)
        b = np.zeros(self.N)
        gi = np.zeros(self.N + 1)
        a = np.zeros(self.N + 1)
        
        for i in range(self.N):
            g_dc = 10**(-3*self.delay[i]/(T_60_DC*self.target_fs))
            g_ny = 10**(-3*self.delay[i]/(T_60_Ny*self.target_fs))
            pi[i] = (g_dc - g_ny)/(g_ny + g_dc)
            gi[i] = g_ny + g_ny*pi[i] 
            b[i] = gi[i]
            if i == 0:
                a[i] = 1
            else:
                a[i] = -pi[i]
        
        return pi, gi
        
    def Process(self, in_signal, absorption):

        pathmin = 5
        d=1/pathmin
        k = np.sqrt(1 - absorption)
        c = k**self.delay/pathmin
        b = 1
        
        if (len(in_signal) != 2):
            in_signal = [in_signal, in_signal]
            
        in_signalR = in_signal[0]
        in_signalL = in_signal[1]
        
        circulant_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            circulant_matrix[i] = np.roll(self.vectorCircular, -i)
            
        yR1, yL1 = 0, 0
        yR2, yL2 = 0, 0
        yR3, yL3 = 0, 0
        yR4, yL4 = 0, 0
        yR5, yL5 = 0, 0
        yR6, yL6 = 0, 0
        yR7, yL7 = 0, 0
        yR8, yL8 = 0, 0
        
        zR1, zL1 = deque([0] * self.delay[0], maxlen=self.delay[0]), deque([0] * self.delay[0], maxlen=self.delay[0])
        zR2, zL2 = deque([0] * self.delay[1], maxlen=self.delay[1]), deque([0] * self.delay[1], maxlen=self.delay[1])
        zR3, zL3 = deque([0] * self.delay[2], maxlen=self.delay[2]), deque([0] * self.delay[2], maxlen=self.delay[2])
        zR4, zL4 = deque([0] * self.delay[3], maxlen=self.delay[3]), deque([0] * self.delay[3], maxlen=self.delay[3])
        zR5, zL5 = deque([0] * self.delay[4], maxlen=self.delay[4]), deque([0] * self.delay[4], maxlen=self.delay[4])
        zR6, zL6 = deque([0] * self.delay[5], maxlen=self.delay[5]), deque([0] * self.delay[5], maxlen=self.delay[5])
        zR7, zL7 = deque([0] * self.delay[6], maxlen=self.delay[6]), deque([0] * self.delay[6], maxlen=self.delay[6])
        zR8, zL8 = deque([0] * self.delay[7], maxlen=self.delay[7]), deque([0] * self.delay[7], maxlen=self.delay[7])


        out_signalR = np.zeros_like(in_signalR)
        out_signalL = np.zeros_like(in_signalL)

        for n in range(len(out_signalR)):
            yR1 = self.pi[0]*yR1 + self.gi[0]*zR1[-1]
            yR2 = self.pi[1]*yR2 + self.gi[1]*zR2[-1]
            yR3 = self.pi[2]*yR3 + self.gi[2]*zR3[-1]
            yR4 = self.pi[3]*yR4 + self.gi[3]*zR4[-1]
            yR5 = self.pi[4]*yR5 + self.gi[4]*zR5[-1]
            yR6 = self.pi[5]*yR6 + self.gi[5]*zR6[-1]
            yR7 = self.pi[6]*yR7 + self.gi[6]*zR7[-1]
            yR8 = self.pi[7]*yR8 + self.gi[7]*zR8[-1]
            
            yL1 = self.pi[0]*yL1 + self.gi[0]*zL1[-1]
            yL2 = self.pi[1]*yL2 + self.gi[1]*zL2[-1]
            yL3 = self.pi[2]*yL3 + self.gi[2]*zL3[-1]
            yL4 = self.pi[3]*yL4 + self.gi[3]*zL4[-1]
            yL5 = self.pi[4]*yL5 + self.gi[4]*zL5[-1]
            yL6 = self.pi[5]*yL6 + self.gi[5]*zL6[-1]
            yL7 = self.pi[6]*yL7 + self.gi[6]*zL7[-1]
            yL8 = self.pi[7]*yL8 + self.gi[7]*zL8[-1]

            tmpR = [yR1, yR2, yR3, yR4, yR5, yR6, yR7, yR8] 
            tmpL = [yL1, yL2, yL3, yL4, yL5, yL6, yL7, yL8] 

            zR1.appendleft(tmpR@circulant_matrix[0].T + in_signalR[n]*b)
            zR2.appendleft(tmpR@circulant_matrix[1].T + in_signalR[n]*b)
            zR3.appendleft(tmpR@circulant_matrix[2].T + in_signalR[n]*b)
            zR4.appendleft(tmpR@circulant_matrix[3].T + in_signalR[n]*b)
            zR5.appendleft(tmpR@circulant_matrix[4].T + in_signalR[n]*b)
            zR6.appendleft(tmpR@circulant_matrix[5].T + in_signalR[n]*b)
            zR7.appendleft(tmpR@circulant_matrix[6].T + in_signalR[n]*b)
            zR8.appendleft(tmpR@circulant_matrix[7].T + in_signalR[n]*b)
            
            zL1.appendleft(tmpL@circulant_matrix[0].T + in_signalL[n]*b)
            zL2.appendleft(tmpL@circulant_matrix[1].T + in_signalL[n]*b)
            zL3.appendleft(tmpL@circulant_matrix[2].T + in_signalL[n]*b)
            zL4.appendleft(tmpL@circulant_matrix[3].T + in_signalL[n]*b)
            zL5.appendleft(tmpL@circulant_matrix[4].T + in_signalL[n]*b)
            zL6.appendleft(tmpL@circulant_matrix[5].T + in_signalL[n]*b)
            zL7.appendleft(tmpL@circulant_matrix[6].T + in_signalL[n]*b)
            zL8.appendleft(tmpL@circulant_matrix[7].T + in_signalL[n]*b)

            out_signalR[n] = d*in_signalR[n] + c[0]*tmpR[0] + c[1]*tmpR[1] + c[2]*tmpR[2] + c[3]*tmpR[3] + c[4]*tmpR[4] + c[5]*tmpR[5] + c[6]*tmpR[6] + c[7]*tmpR[7]
            out_signalL[n] = d*in_signalL[n] + c[0]*tmpL[0] + c[1]*tmpL[1] + c[2]*tmpL[2] + c[3]*tmpL[3] + c[4]*tmpL[4] + c[5]*tmpL[5] + c[6]*tmpL[6] + c[7]*tmpL[7]
            
        return out_signalL, out_signalR
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        