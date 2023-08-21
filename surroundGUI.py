import sys, os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import SurroundSound as spatial
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QIcon, 
    QPixmap)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QComboBox,
    QStackedLayout,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFormLayout,
    QFileDialog,
    QWidget
    )

WINDOW_SIZEW = 400
WINDOW_SIZEH = 450
BUTTON_SIZE = 40


class PyHRTFWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surround Sound")
        self.setFixedSize(WINDOW_SIZEW, WINDOW_SIZEH)
        # Create a general layout
        self.generalLayout = QVBoxLayout()
        
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
                
        self.setCentralWidget(centralWidget)
        self._sectionAudio()
        self._sectionHRTF()
        self._sectionPlay()
        self.target_fs = int(48e3)
        self.audio = None
        self.setAzimuth_51 = [0, 0, 30, 110, 250, 330]
        self.setElevation_51 = [0, 0, 0, 45, 45, 0]
        self.absorption_51 = [0.50, 0.50, 0.63, 0.75, 0.73, 0.65]
        self.setAzimuth_71 = [0, 0, 30, 90, 135, 225, 270, 330]
        self.setElevation_71 = [0, 0, 0, 45, 45, 45, 45, 0]
        self.absorption_71 = [0.50, 0.50, 0.63, 0.7, 0.75, 0.73, 0.69, 0.65]
        self.myReverb = spatial._Reverb_(self.target_fs)
        self.Sub_H = None
        self.Center_H = None 
        self.FrontL_H = None 
        self.FrontR_H = None
        self.RearL_H = None
        self.RearR_H = None
        self.SurroundL_H = None
        self.SurroundR_H = None
        
    def _sectionAudio(self):
        self.titleAudio = QLabel()
        self.titleAudio.setText("Audio Section")
        self.titleAudio.setAlignment(Qt.AlignmentFlag.AlignCenter)
       
        # Create a Layout for Audio
        self.generalAudio = QGridLayout()
        self.loadAudio = QLabel()
        self.loadAudio.setText("Load Audio:")
        self.generalAudio.addWidget(self.loadAudio, 0, 0)
        self.searchAudio = QPushButton("Select", self)
        self.generalAudio.addWidget(self.searchAudio , 0, 1)
        self.searchAudio.clicked.connect( self.openFileWav)
        
        self.generalLayout.addWidget(self.titleAudio)
        self.generalLayout.addLayout(self.generalAudio)
        
        
    def openFileWav(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileAudio, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo .wav", "", "Archivos WAV (*.wav)", options=options)

        if fileAudio:
            base_audio = os.path.basename(fileAudio)
            self.searchAudio.setText(base_audio)
            self.audio = spatial.Audio(fileAudio, self.target_fs)
        
    def _sectionHRTF(self):
        
        self.titleHRTF = QLabel()
        self.titleHRTF.setText("HRTF Section")
        self.titleHRTF.setAlignment(Qt.AlignmentFlag.AlignCenter)
       
        # Create a Layout for HRTF
        self.generalHRTF = QGridLayout()
        self.loadHRTF = QLabel()
        self.loadHRTF.setText("Load HRTF:")
        self.generalHRTF.addWidget(self.loadHRTF, 0, 0)
        
        # Create button click information
        self.searchHRTF = QPushButton("Select", self)
        self.generalHRTF.addWidget(self.searchHRTF, 0, 1)
        self.searchHRTF.clicked.connect(self.openFileHRTF)
    
        # Create pop up menu
        self.configHRTF = QLabel()
        self.configHRTF.setText("Configuration:")
        self.generalHRTF.addWidget(self.configHRTF, 1, 0)
        listCombo = ["5.1", "7.1"]
        self.configCombo = QComboBox()
        self.configCombo.addItems(listCombo)
        self.configCombo.activated.connect(self.switchPage)
        
        # Create the stacked layout
        self.stackedLayout = QStackedLayout()
        
        self.testHRTF = QLabel()
        self.testHRTF.setText("Test:")

        # Configuration 51
        self.imageHuman = QLabel()
        pictureHuman = QPixmap('picture.jpeg')
        self.imageHuman.setPixmap(pictureHuman)
        self.imageHuman.setScaledContents(True)
        self.resize(pictureHuman.width(), pictureHuman.height())
        self.speaker51 = QWidget()
        self.speaker51layout = QGridLayout()
        
        self.FL5 = QPushButton("FL", self)
        self.speaker51layout.addWidget(self.FL5, 0, 0)
        self.FL5.clicked.connect(lambda: self.generateWav51('FL'))
        
        self.C5 = QPushButton("C", self)
        self.speaker51layout.addWidget(self.C5, 0, 1)
        self.C5.clicked.connect(lambda: self.generateWav51('C'))
        
        self.FR5 = QPushButton("FR", self)
        self.speaker51layout.addWidget(self.FR5, 0, 2)
        self.FR5.clicked.connect(lambda: self.generateWav51('FR'))
        
        self.speaker51layout.addWidget(self.imageHuman, 1, 1, 3, 1)
        
        self.RR5 = QPushButton("RL", self)
        self.speaker51layout.addWidget(self.RR5, 4, 0)
        self.RR5.clicked.connect(lambda: self.generateWav51('RR'))
        
        self.RL5 = QPushButton("RR", self)
        self.speaker51layout.addWidget(self.RL5, 4, 2)
        self.RL5.clicked.connect(lambda: self.generateWav51('RL'))
        
        self.speaker51.setLayout(self.speaker51layout)
        self.stackedLayout.addWidget(self.speaker51)
        
        # Configuration 71
        self.imageHuman7 = QLabel()
        pictureHuman2 = QPixmap('picture.jpeg')
        self.imageHuman7.setPixmap(pictureHuman2)
        self.imageHuman7.setScaledContents(True)
        self.resize(pictureHuman2.width(), pictureHuman2.height())
        self.speaker71 = QWidget()
        self.speaker71layout = QGridLayout()
        
        self.FL7 = QPushButton("FL", self)
        self.speaker71layout.addWidget(self.FL7, 0, 0)
        self.FL7.clicked.connect(lambda: self.generateWav71('FL'))
        
        self.C7 = QPushButton("C", self)
        self.speaker71layout.addWidget(self.C7, 0, 2)
        self.C7.clicked.connect(lambda: self.generateWav71('C'))
        
        self.FR7 = QPushButton("FR", self)
        self.speaker71layout.addWidget(self.FR7, 0, 4)
        self.FR7.clicked.connect(lambda: self.generateWav71('FR'))
        
        self.SL7 = QPushButton("SL", self)
        self.speaker71layout.addWidget(self.SL7, 1, 0)
        self.SL7.clicked.connect(lambda: self.generateWav71('SL'))
        
        self.speaker71layout.addWidget(self.imageHuman7, 1, 1, 1, 3)
        
        self.SR7 = QPushButton("SR", self)
        self.speaker71layout.addWidget(self.SR7, 1, 4)
        self.SR7.clicked.connect(lambda: self.generateWav71('SR'))
        
        self.RR7 = QPushButton("RL", self)
        self.speaker71layout.addWidget(self.RR7, 2, 1)
        self.RR7.clicked.connect(lambda: self.generateWav71('RR'))
        
        self.RL7 = QPushButton("RR", self)
        self.speaker71layout.addWidget(self.RL7, 2, 3)
        self.RL7.clicked.connect(lambda: self.generateWav71('RL'))
        
        self.speaker71.setLayout(self.speaker71layout)
        self.stackedLayout.addWidget(self.speaker71)
        
        self.generalHRTF.addWidget(self.configCombo, 1, 1)
        self.generalHRTF.addWidget(self.testHRTF, 2, 0)
        self.generalHRTF.addLayout(self.stackedLayout, 2, 1)
        self.generalLayout.addWidget(self.titleHRTF)
        self.generalLayout.addLayout(self.generalHRTF)
        
    def openFileHRTF(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileHRTF, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo .sofa", "", "Archivos SOFA (*.sofa)", options=options)
        if fileHRTF:
            base_hrtf = os.path.basename(fileHRTF)
            self.searchHRTF.setText(base_hrtf)
            self.myHRTF = spatial.oneHRTF(fileHRTF, self.target_fs)
        
    def switchPage(self):
        self.stackedLayout.setCurrentIndex(self.configCombo.currentIndex())
        
    def _sectionPlay(self):
        self.playButtom = QPushButton("Play",self)
        self.playButtom.setFixedSize(150,30)
        self.playButtom.clicked.connect(lambda: self._playAll())
        self.generalLayout.setAlignment(Qt.AlignCenter)
        self.generalLayout.addWidget(self.playButtom, alignment=Qt.AlignRight)
        
    def calculate_parameters(self):
        
        ''' Crossover '''
        LP_Sub = spatial.CrossoverFilter("LR", 2, "low", 120, True, self.target_fs)
        HP_Front = spatial.CrossoverFilter("LR", 2, "high", 50, True, self.target_fs)
        HP_Rear = spatial.CrossoverFilter("LR", 2, "high", 80, True, self.target_fs)
        HP_Surround = spatial.CrossoverFilter("LR", 2, "high", 50, True, self.target_fs)
        HP_Center = spatial.CrossoverFilter("LR", 2, "high", 60, True, self.target_fs)
        
        # Subwoofer
        Sub = signal.lfilter(LP_Sub.b, LP_Sub.a, self.audio.mono)
        
        # Center
        Center = signal.lfilter(HP_Center.b, HP_Center.a, self.audio.mono)
        
        # Front
        FrontL = signal.lfilter(HP_Front.b, HP_Front.a, self.audio.xtmp)
        FrontR = signal.lfilter(HP_Front.b, HP_Front.a, self.audio.ytmp)
        
        # Rear
        RearL = signal.lfilter(HP_Rear.b, HP_Rear.a, self.audio.xtmp)
        RearR = signal.lfilter(HP_Rear.b, HP_Rear.a, self.audio.ytmp)
        
        # Surround
        SurroundL = signal.lfilter(HP_Surround.b, HP_Surround.a, self.audio.xtmp)
        SurroundR = signal.lfilter(HP_Surround.b, HP_Surround.a, self.audio.ytmp)
        
        ''' Loudspeaker Simulation '''
        woofer = spatial.Loudspeaker(Re = 3.8, Le = 8.5e-3, Bl = 38.3, fr = 26.3, Mms = 676e-3, Kms = 54e-6, Qms = 7.9, Sd = 152.5e-3, fs = self.target_fs)
        screen = spatial.Loudspeaker(Re = 6, Le = 0.75e-3, Bl = 7.5, fr = 45.5, Mms = 10.6e-3, Kms = 1.15e-3, Qms = 3.61, Sd = 21.64e-3, fs = self.target_fs)
        surround = spatial.Loudspeaker(Re = 6.4, Le = 0.63e-3, Bl = 6.82, fr = 48.2, Mms = 7e-3, Kms = 1.05e-3, Qms = 3.64, Sd = 13.27e-3, fs = self.target_fs)
        
        self.Sub_H = signal.lfilter(woofer.b, woofer.a, Sub)
        
        self.Center_H = signal.lfilter(screen.b, screen.a, Center)
        
        self.FrontL_H = signal.lfilter(screen.b, screen.a, FrontL)
        self.FrontR_H = signal.lfilter(screen.b, screen.a, FrontR)
        
        self.RearL_H = signal.lfilter(surround.b, surround.a, RearL)
        self.RearR_H = signal.lfilter(surround.b, surround.a, RearR)
        
        self.SurroundL_H = signal.lfilter(surround.b, surround.a, SurroundL)
        self.SurroundR_H = signal.lfilter(surround.b, surround.a, SurroundR)
                
        
    def generateWav51(self, parameter):
        
        
        if self.Sub_H is None or self.Center_H is None or self.FrontL_H is None or self.FrontR_H is None or self.RearL_H is None or self.RearR_H is None:
            # Calculate the parameters if they haven't been calculated 
            self.calculate_parameters()
            
    
        stereo51 = np.zeros([10, 2])
        
        if parameter == "C":
            rend_L, rend_R = self.hrtf.one_position_configuration(0, 0, self.Center_H)
            reverb_L, reverb_R = self.myReverb.Process(self.Center_H, self.absorption_51[1])
            
        elif parameter == "FR":
            rend_L, rend_R = self.hrtf.one_position_configuration(30, 0, [self.FrontL_H, self.FrontR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_51[2])
            
        elif parameter == "FL":
            rend_L, rend_R = self.hrtf.one_position_configuration(330, 0, [self.FrontL_H, self.FrontR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_51[3])
            
        elif parameter == "RR":
            rend_L, rend_R = self.hrtf.one_position_configuration(110, 45, [self.RearL_H, self.RearR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.RearL_H, self.RearR_H], self.absorption_51[4])
            
        elif parameter == "RL":
            rend_L, rend_R = self.hrtf.one_position_configuration(250, 45, [self.RearL_H, self.RearR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.RearL_H, self.RearR_H], self.absorption_51[5])
        
        tot_L = rend_L + reverb_L
        tot_R = rend_R + reverb_R
        
        M = np.max([np.abs(tot_L), np.abs(tot_R)]) 
        if len(stereo51) < len(tot_L):
            diff = len(tot_L) - len(stereo51)
            stereo51 = np.append(stereo51, np.zeros([diff,2]),0)
            
        stereo51[0:len(tot_L),0] += (tot_L/M)
        stereo51[0:len(tot_R),1] += (tot_R/M)
        
        file_name = f"test51{parameter}.wav"
        
        sf.write(file_name, stereo51, self.target_fs)
        
        print(f"Archivo guardado: {file_name}")
        
    def generateWav71(self, parameter):
        
        if self.Sub_H is None or self.Center_H is None or self.FrontL_H is None or self.FrontR_H is None or self.RearL_H is None or self.RearR_H is None or self.SurroundL_H is None or self.SurroundR_H is None: 
            # Calculate the parameters if they haven't been calculated 
            self.calculate_parameters()
    
        stereo71 = np.zeros([10, 2])
        
        if parameter == "C":
            rend_L, rend_R = self.hrtf.one_position_configuration(0, 0, self.Center_H)
            reverb_L, reverb_R = self.myReverb.Process(self.Center_H, self.absorption_71[1])
            
        elif parameter == "FR":
            rend_L, rend_R = self.hrtf.one_position_configuration(30, 0, [self.FrontL_H, self.FrontR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_71[2])
            
        elif parameter == "FL":
            rend_L, rend_R = self.hrtf.one_position_configuration(330, 0, [self.FrontL_H, self.FrontR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_71[3])
            
        elif parameter == "SR":
            rend_L, rend_R = self.hrtf.one_position_configuration(90, 0, [self.SurroundL_H, self.SurroundR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_71[4])
            
        elif parameter == "SL":
            rend_L, rend_R = self.hrtf.one_position_configuration(270, 0, [self.SurroundL_H, self.SurroundR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.FrontL_H, self.FrontR_H], self.absorption_71[5])
            
        elif parameter == "RR":
            rend_L, rend_R = self.hrtf.one_position_configuration(110, 45, [self.RearL_H, self.RearR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.RearL_H, self.RearR_H], self.absorption_71[6])
            
        elif parameter == "RL":
            rend_L, rend_R = self.hrtf.one_position_configuration(250, 45, [self.RearL_H, self.RearR_H])
            reverb_L, reverb_R = self.myReverb.Process([self.RearL_H, self.RearR_H], self.absorption_71[7])
        
        tot_L = rend_L + reverb_L
        tot_R = rend_R + reverb_R
        
        M = np.max([np.abs(tot_L), np.abs(tot_R)]) 
        if len(stereo71) < len(tot_L):
            diff = len(tot_L) - len(stereo71)
            stereo71 = np.append(stereo71, np.zeros([diff,2]),0)
            
        stereo71[0:len(tot_L),0] += (tot_L/M)
        stereo71[0:len(tot_R),1] += (tot_R/M)
        
        file_name = f"test71{parameter}.wav"
        
        sf.write(file_name, stereo71, self.target_fs)
        
        print(f"Archivo guardado: {file_name}")
        
    def _playAll(self):
        
        if self.Sub_H is None or self.Center_H is None or self.FrontL_H is None or self.FrontR_H is None or self.RearL_H is None or self.RearR_H is None or self.SurroundL_H is None or self.SurroundR_H is None: 
            # Calculate the parameters if they haven't been calculated 
            self.calculate_parameters()
            
        config_index = self.configCombo.currentIndex()
        
        if config_index == 0:
            input_signal = [self.Sub_H, self.Center_H, [self.FrontL_H, self.FrontR_H], [self.RearL_H, self.RearR_H], [self.RearL_H, self.RearR_H], [self.FrontL_H, self.FrontR_H]]

        elif config_index == 1:
            input_signal = [self.Sub_H, self.Center_H, [self.FrontL_H, self.FrontR_H], [self.SurroundL_H, self.SurroundR_H], [self.RearL_H, self.RearR_H], [self.RearL_H, self.RearR_H], [self.SurroundL_H, self.SurroundR_H], [self.FrontL_H, self.FrontR_H]]

        
        stereo3D = np.zeros([10, 2]) # Specialized Sources

        for m, in_signal in enumerate(input_signal):
            
            if config_index == 0:
                azimuth, elevation, absorption = self.setAzimuth_51[m], self.setElevation_51[m], self.absorption_51[m]
            if config_index == 1:
                azimuth, elevation, absorption = self.setAzimuth_71[m], self.setElevation_71[m], self.absorption_71[m]
                     
            rend_L, rend_R = self.myHRTF.one_position_configuration(azimuth, elevation, in_signal)
            reverb_L, reverb_R = self.myReverb.Process(in_signal, absorption)
            
            tot_L = rend_L + reverb_L
            tot_R = rend_R + reverb_R
            
            #M = np.max([np.abs(tot_L), np.abs(tot_R)]) 
            if len(stereo3D) < len(tot_L):
                diff = len(tot_L) - len(stereo3D)
                stereo3D = np.append(stereo3D, np.zeros([diff,2]),0)
                
            stereo3D[0:len(tot_L),0] += (tot_L)
            stereo3D[0:len(tot_R),1] += (tot_R)
            
        M = np.max([stereo3D[:,0], stereo3D[:,1]]) 
        
        stereo3D /= M
        
        if config_index == 0:
            file_name = "SpatialAudio51.wav"
        elif config_index == 1:
            file_name = "SpatialAudio71.wav"
        
        sf.write(file_name, stereo3D, self.target_fs)
        
        print(f"Archivo guardado: {file_name}")

        
def main():
    pyHRTFApp = QApplication(sys.argv)
    pyHRTFWindow = PyHRTFWindow()
    pyHRTFWindow.show()
    sys.exit(pyHRTFApp.exec_())
    
if __name__ == "__main__":
    main()