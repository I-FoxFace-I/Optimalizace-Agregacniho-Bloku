
import gurobipy as g
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import testData as gen
import tensorflow as tf
import statsmodels.api as sm

from cProfile import label
from re import A
from aFRR_test import aFRR_Generator
from mFRR_test import mFRR_Generator
from lstm import LSTM_Predictor
from gurobipy import GRB, quicksum, Model, abs_
from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter, freqz
from scipy.signal import lfilter, hamming
from librosa import lpc
from my_colors import cnames
from statsmodels.graphics import tsaplots


class My_Model:
    """
    Trida reprezentujici MILP model, ktery nalezne optimalni rozlozeni vykonu of VPP slozene z Vyrobnic jednotek a Akumulatoru
    """
    def __init__(self):
        """
        Constructor definujici promenne a parametry MILP modelu
        """
        #Model
        self.VPP_Model = g.Model()
        #Promenne
        self.N_t = 0                         # Pocet casovych okamziku
        self.N_P = 0                         # Pocet vyrobnich jednotek
        self.N_S = 0                         # Pocet akumulatru
        self.Time_Stamp = 0                  #
        self.P_i = list()                    # Cinny vykon vyrobni jednotky i            (FLOAT)   
        self.SOC_i = list()                  # Stav nabiti akumulatoru i                 (FLOAT)
        self.x_i = list()                    # Je zapnuta vyrobni jednotka i             (BOOLEAN)   
        self.y_a = list()                    # Aktivace vyrobni jednotky i               (BOOLEAN)
        self.y_d = list()                    # Deaktivace vyrobni jednotky i             (BOOLEAN)  
        self.P_c = list()                    # Nabijeci vykon akumulatoru i              (FLOAT)   
        self.P_d = list()                    # Vybijeci vykon akumulatoru i              (FLOAT)
        self.c_i = list()                    # Je akumulator i nabijen                   (BOOLEAN)  
        self.d_i = list()                    # Je akumulator i vybijen                   (BOOLEAN)  
        self.P_g = list()                    # Vykon odebirany ze site                   (FLOAT)
        #Parametery                          # 
        self.aFRR = list()                   # Hodnoty pro aFRR                          (Dictionary)
        self.mFRR = list()                   # Hodnoty pro mFRR                          (Dictionary)
        self.P_DG = list()                   # Hodnoty pro Diagramovy vykon              (FLOAT)
        self.PP = list()                     # Seznam parametru vyrobnich jednotek       (Dictionary)
        self.ST = list()                     # Seznam parametru akumulatoru              (Dictionary)
        self.Grid_price = list()             # Aktualni cena elektriny odebirane ze site (FLOAT)
        self.Epsion = 0                      # Tolerovana odchylka                       (FLOAT)  
        self.Init_Conditions = dict()        # Pocatecni podminky

        #Output Buffer
        self.OUT_P_i = np.zeros((4,480))
        self.OUT_P_c = np.zeros((4,480))
        self.OUT_P_d = np.zeros((4,480))
        self.t_act = 0
        self.cum_sum = 0


    
    def load_params(self,PP_params,ST_params,P_DG,aFRR,mFRR,Grid_price,Time_Stamp):
        """
        Nastavi parametery modelu

        :parametr PP_params:   dictionary obsahujici parametry vyrobnich jednotek
        :parametr ST_params:   dictionary obsahujici parametry akumulatoru
        :parametr P_DG:        list s pozadavky  Diagramoveho vykony
        :parametr aFRR:        dictionary obsahujici informace o aFRR
        :parametr aFRR:        dictionary obsahujici informace o mFRR
        :parametr grid:        list s cenami elektricke energie v danem case
        """
        self.P_DG = P_DG
        self.aFRR = aFRR
        self.mFRR = mFRR
        self.PP = PP_params
        self.ST = ST_params
        self.Grid_price = Grid_price
        self.Time_Stamp = Time_Stamp
        self.N_t = len(P_DG)
        self.N_P = len(PP_params)
        self.N_S = len(ST_params)
        self.Epsilon = 0.1

    def create_variables(self):
        """
        Funkce vytvori promenne  of MILP model
        """
        self.VPP_Model = g.Model()
        # Vyrobni jednotky
        self.P_i = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.CONTINUOUS)
        self.x_i = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.BINARY)
        self.y_a = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.BINARY)
        self.y_d = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.BINARY)

        # Akumulatory
        self.SOC_i = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        self.P_c = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        self.P_d = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        self.c_i = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.BINARY)
        self.d_i = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.BINARY)

        # Vykon ze site
        self.P_g = self.VPP_Model.addVars(self.N_t,vtype=g.GRB.CONTINUOUS)
    
    def add_constrains(self):
        """
        Funkce definuje omezeni MILP modelu
        """

        #################
        #Init conditions#
        #################
        for t in range(self.Time_Stamp):
            #Power plants
            for k in range(self.N_P):
                if(k>12):
                    self.VPP_Model.addConstr(self.P_i[k,t] == self.Init_Conditions['P_i'][t][k])
                    self.VPP_Model.addConstr(self.x_i[k,t] == self.Init_Conditions['x_i'][t][k])
                    self.VPP_Model.addConstr(self.y_a[k,t] == self.Init_Conditions['y_a'][t][k])
                    self.VPP_Model.addConstr(self.y_d[k,t] == self.Init_Conditions['y_d'][t][k])
                else:
                    self.VPP_Model.addConstr(self.P_i[k,t] == self.Init_Conditions['P_i'][t][k])
                    self.VPP_Model.addConstr(self.x_i[k,t] == 1)
                    self.VPP_Model.addConstr(self.y_a[k,t] == 0)
                    self.VPP_Model.addConstr(self.y_d[k,t] == 0)
            
            #Storage systems
            for k in range(self.N_S):
                self.VPP_Model.addConstr(self.SOC_i[k,t] == self.Init_Conditions['SOC_i'][t][k])
                self.VPP_Model.addConstr(self.P_c[k,t] == self.Init_Conditions['P_c'][t][k])
                self.VPP_Model.addConstr(self.P_d[k,t] == self.Init_Conditions['P_d'][t][k])
                self.VPP_Model.addConstr(self.c_i[k,t] == self.Init_Conditions['c_i'][t][k])
                self.VPP_Model.addConstr(self.d_i[k,t] == self.Init_Conditions['d_i'][t][k])
            
            #Grid price
            self.VPP_Model.addConstr(self.P_g[t] == self.Init_Conditions['P_g'][t])

        ############
        #Constrains#
        ############
        for t in range(self.Time_Stamp, self.N_t):
            #Vyrobni jednotky
            for k in range(self.N_P):
                self.VPP_Model.addConstr(self.P_i[k,t] >= self.x_i[k,t]*self.PP[k]['P_MIN'])
                self.VPP_Model.addConstr(self.P_i[k,t] <= self.x_i[k,t]*self.PP[k]['P_MAX'])
                if(t>0):
                    self.VPP_Model.addConstr(self.P_i[k,t] - self.P_i[k,t-1] <= self.PP[k]['T'])
                    self.VPP_Model.addConstr(self.P_i[k,t] - self.P_i[k,t-1] >= -self.PP[k]['T'])
                    self.VPP_Model.addConstr(self.x_i[k,t] - self.x_i[k,t-1] == self.y_a[k,t])
                    self.VPP_Model.addConstr(self.x_i[k,t-1] - self.x_i[k,t] == self.y_d[k,t])
                    self.VPP_Model.addConstr(self.y_d[k,t] + self.y_a[k,t] <=1)

            #Akumulatory
            for k in range(self.N_S):
                self.VPP_Model.addConstr(self.SOC_i[k,t] >= self.ST[k]['SOC_MIN'])
                self.VPP_Model.addConstr(self.SOC_i[k,t] <= self.ST[k]['SOC_MAX'])
                if(t>0):
                    self.VPP_Model.addConstr(self.SOC_i[k,t-1] + self.ST[k]['eta']*self.P_c[k,t] - (self.P_d[k,t])/self.ST[k]['eta']==self.SOC_i[k,t])
                else:
                    self.VPP_Model.addConstr(self.SOC_i[k,t] == self.ST[k]['SOC_MIN']+(self.ST[k]['SOC_MAX']-self.ST[k]['SOC_MIN'])/2)
                self.VPP_Model.addConstr(self.P_c[k,t] >= self.c_i[k,t]*self.ST[k]['PC_MIN'])
                self.VPP_Model.addConstr(self.P_c[k,t] <= self.c_i[k,t]*self.ST[k]['PC_MAX'])
                self.VPP_Model.addConstr(self.P_d[k,t] >= self.d_i[k,t]*self.ST[k]['PD_MIN'])
                self.VPP_Model.addConstr(self.P_d[k,t] <= self.d_i[k,t]*self.ST[k]['PD_MAX'])
                if(t>0):
                    self.VPP_Model.addConstr(self.P_c[k,t] - self.P_c[k,t-1] <= self.c_i[k,t]*self.ST[k]['TC'])
                    self.VPP_Model.addConstr(self.P_c[k,t] - self.P_c[k,t-1] >= -self.c_i[k,t]*self.ST[k]['TC'])
                    self.VPP_Model.addConstr(self.P_d[k,t] - self.P_d[k,t-1] <= self.d_i[k,t]*self.ST[k]['TD'])
                    self.VPP_Model.addConstr(self.P_d[k,t] - self.P_d[k,t-1] >= -self.d_i[k,t]*self.ST[k]['TD'])
                self.VPP_Model.addConstr(self.c_i[k,t] + self.d_i[k,t] <= 1)

            #Rovnovaha v siti
            P_i = g.quicksum(self.P_i[k,t] for k in range(self.N_P))
            P_d = g.quicksum(self.P_d[k,t]*(1/self.ST[k]['eta']) for k in range(self.N_S))
            P_c = g.quicksum(self.P_c[k,t]*(self.ST[k]['eta']) for k in range(self.N_S))
            PU = g.quicksum(self.x_i[k,t]*self.PP[k]['P_MAX'] for k in range(self.N_P))
            PL = g.quicksum(self.x_i[k,t]*self.PP[k]['P_MIN'] for k in range(self.N_P))
            PUD = g.quicksum(self.ST[k]['PD_MAX']*(1/self.ST[k]['eta']) for k in range(self.N_S))
            PLC = g.quicksum(self.ST[k]['PC_MAX']*(self.ST[k]['eta']) for k in range(self.N_S))
            
            self.VPP_Model.addConstr(self.P_g[t] >= 0)
            # Dodrzeni rovnovahy v siti
            # Uzavreni do intervalu
            self.VPP_Model.addConstr(P_i + P_d - P_c - self.P_g[t] <= self.P_DG[t] + self.aFRR[t]['A+'] + self.mFRR[t]['A+'] + self.Epsilon)
            self.VPP_Model.addConstr(P_i + P_d - P_c - self.P_g[t] >= self.P_DG[t] + self.aFRR[t]['A-'] + self.mFRR[t]['A-'] - self.Epsilon)
            # Zajisteni aktivace zdroju pro pripadne poskytovani aFRR a mFRR
            self.VPP_Model.addConstr(PU + PUD >= self.P_DG[t] + self.aFRR[t]['R+'] + self.mFRR[t]['R+'])
            self.VPP_Model.addConstr(PL - PLC <= self.P_DG[t] + self.aFRR[t]['R-'] + self.mFRR[t]['R-'])

    def set_objective(self):
        """
        Funkce definuje kriterialni funkci MILP modelu
        """
        
        #Pridavame absolutni hodnotu
        big_M = 10000000000000
        y_P = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.BINARY)
        y_SOC = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.BINARY)
        abs_P = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.CONTINUOUS)
        abs_SOC = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        tmpvar_PP = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.CONTINUOUS)
        tmpvar_PM = self.VPP_Model.addVars(self.N_P,self.N_t,vtype=g.GRB.CONTINUOUS)
        tmpvar_SOCP = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        tmpvar_SOCM = self.VPP_Model.addVars(self.N_S,self.N_t,vtype=g.GRB.CONTINUOUS)
        for t in range(self.N_t):
            # Power plants
            for k in range(self.N_P):
                self.VPP_Model.addConstr(tmpvar_PP[k,t] <= self.P_i[k,t] - (self.PP[k]['P_MIN']+(self.PP[k]['P_MAX']-self.PP[k]['P_MIN'])/2) + big_M*y_P[k,t])
                self.VPP_Model.addConstr(tmpvar_PP[k,t] >= self.P_i[k,t] - (self.PP[k]['P_MIN']+(self.PP[k]['P_MAX']-self.PP[k]['P_MIN'])/2) - big_M*y_P[k,t])
                self.VPP_Model.addConstr(tmpvar_PP[k,t] <=  0 + big_M*(1-y_P[k,t]))
                self.VPP_Model.addConstr(tmpvar_PP[k,t] >=  0 - big_M*(1-y_P[k,t]))
                self.VPP_Model.addConstr(tmpvar_PM[k,t] <= 0 + big_M*y_P[k,t])
                self.VPP_Model.addConstr(tmpvar_PM[k,t] >= 0 - big_M*y_P[k,t])
                self.VPP_Model.addConstr(tmpvar_PM[k,t] <=  -1*self.P_i[k,t] + (self.PP[k]['P_MIN']+(self.PP[k]['P_MAX']-self.PP[k]['P_MIN'])/2) + big_M*(1-y_P[k,t]))
                self.VPP_Model.addConstr(tmpvar_PM[k,t] >=  -1*self.P_i[k,t] + (self.PP[k]['P_MIN']+(self.PP[k]['P_MAX']-self.PP[k]['P_MIN'])/2) - big_M*(1-y_P[k,t]))
                self.VPP_Model.addConstr(abs_P[k,t] == tmpvar_PP[k,t] + tmpvar_PM[k,t])
            # Storage systems
            for k in range(self.N_S):
                self.VPP_Model.addConstr(tmpvar_SOCP[k,t] <= self.SOC_i[k,t] - (self.ST[k]['SOC_MIN']+(self.ST[k]['SOC_MAX']-self.ST[k]['SOC_MIN'])/2) + big_M*y_SOC[k,t])
                self.VPP_Model.addConstr(tmpvar_SOCP[k,t] >= self.SOC_i[k,t] - (self.ST[k]['SOC_MIN']+(self.ST[k]['SOC_MAX']-self.ST[k]['SOC_MIN'])/2) - big_M*y_SOC[k,t])
                self.VPP_Model.addConstr(tmpvar_SOCP[k,t] <=  0 + big_M*(1-y_SOC[k,t]))
                self.VPP_Model.addConstr(tmpvar_SOCP[k,t] >=  0 - big_M*(1-y_SOC[k,t]))
                self.VPP_Model.addConstr(tmpvar_SOCM[k,t] <= 0 + big_M*y_SOC[k,t])
                self.VPP_Model.addConstr(tmpvar_SOCM[k,t] >= 0 - big_M*y_SOC[k,t])
                self.VPP_Model.addConstr(tmpvar_SOCM[k,t] <=  -1*self.SOC_i[k,t] + (self.ST[k]['SOC_MIN']+(self.ST[k]['SOC_MAX']-self.ST[k]['SOC_MIN'])/2) + big_M*(1-y_SOC[k,t]))
                self.VPP_Model.addConstr(tmpvar_SOCM[k,t] >=  -1*self.SOC_i[k,t] + (self.ST[k]['SOC_MIN']+(self.ST[k]['SOC_MAX']-self.ST[k]['SOC_MIN'])/2) - big_M*(1-y_SOC[k,t]))
                self.VPP_Model.addConstr(abs_SOC[k,t] == tmpvar_SOCP[k,t] + tmpvar_SOCM[k,t])


        #Pomocne promenne pro optimalizaci
        C_PA = g.quicksum(self.y_a[k,t]*self.PP[k]['C_A'] for k in range(self.N_P) for t in range(self.N_t))
        C_PD = g.quicksum(self.y_d[k,t]*self.PP[k]['C_D'] for k in range(self.N_P) for t in range(self.N_t))
        C_PC = g.quicksum(self.x_i[k,t]*self.PP[k]['A'] + self.P_i[k,t]*self.PP[k]['B'] for k in range(self.N_P) for t in range(self.N_t))
        C_SD = g.quicksum(self.d_i[k,t]*self.ST[k]['AD'] + self.P_d[k,t]*self.ST[k]['BD'] for k in range(self.N_S) for t in range(self.N_t))
        C_SC = g.quicksum(self.c_i[k,t]*self.ST[k]['AC'] + self.P_d[k,t]*self.ST[k]['BC'] for k in range(self.N_S) for t in range(self.N_t))
        C_G = g.quicksum(self.P_g[t]*self.Grid_price[t] for t in range(self.N_t))
        C_ABS_SOC = g.quicksum(abs_SOC[k,t]*1.5 for k in range(self.N_S) for t in range(self.N_t))
        C_ABS_P = g.quicksum(abs_P[k,t]*0.5 for k in range(self.N_P) for t in range(self.N_t))

        self.VPP_Model.setObjective(C_PA + C_PD + C_PC + C_SD + C_SC + C_G + C_ABS_SOC + C_ABS_P ,sense = g.GRB.MINIMIZE)

    def calculate(self):
        '''
        Funkce spousti ILP branch and bound optimalizaci
        '''
        self.VPP_Model.optimize()
        print("Actual time: " + str(self.t_act))


    def change_Time(self,Time_Stamp):
        '''
        Pomocna funkce na zmenu velikosti casoveho okna
        '''
        self.Time_Stamp = int(Time_Stamp)

    def save_init_cond(self):
        '''
        Algoritmus si ulozi prvnich T hodnot jako pocatecni podminky ktere vyuzije v nasledujcim vypoctu
        '''
        self.Init_Conditions['P_i'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['x_i'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['y_a'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['y_d'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['SOC_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_c'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_d'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['c_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['d_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_g'] = [0]*self.Time_Stamp

        for t in range(1,self.Time_Stamp+1):
            #Power plants
            for k in range(self.N_P):
                self.Init_Conditions['P_i'][t-1][k] = self.P_i[k,t].x
                self.Init_Conditions['x_i'][t-1][k] = 1 if (self.x_i[k,t].x > 0.75) else 0
                self.Init_Conditions['y_a'][t-1][k] = 1 if (self.y_a[k,t].x>0.75) else 0
                self.Init_Conditions['y_d'][t-1][k] = 1 if (self.y_d[k,t].x>0.75) else 0
                self.OUT_P_i[k][self.t_act] = self.P_i[k,self.Time_Stamp].x
                # Je potreba si zaznamenat aktualni hodnotu vykonu pro kterou pocitame optimalizaci
                if(t==1):
                    self.cum_sum+= round(self.y_a[k,self.Time_Stamp-1].x)*self.PP[k]['C_A'] + round(self.y_d[k,self.Time_Stamp-1].x)*self.PP[k]['C_D'] + round(self.x_i[k,self.Time_Stamp-1].x)*self.PP[k]['A'] + self.P_i[k,self.Time_Stamp-1].x*self.PP[k]['B']
            
            #Storage systems
            for k in range(self.N_S):
                self.Init_Conditions['SOC_i'][t-1][k] = self.SOC_i[k,t].x
                self.Init_Conditions['P_c'][t-1][k] = self.P_c[k,t].x
                self.Init_Conditions['P_d'][t-1][k] = self.P_d[k,t].x
                self.Init_Conditions['c_i'][t-1][k] = 1 if(self.c_i[k,t].x>0.75) else 0
                self.Init_Conditions['d_i'][t-1][k] = 1 if(self.d_i[k,t].x>0.75) else 0
                self.OUT_P_c[k][self.t_act] = self.P_c[k,self.Time_Stamp].x
                self.OUT_P_d[k][self.t_act] = self.P_d[k,self.Time_Stamp].x
                # Je potreba si zaznamenat aktualni hodnotu vykonu pro kterou pocitame optimalizaci
                if t==1:
                    self.OUT_P_c[k][self.t_act] = self.P_c[k,self.Time_Stamp-1].x
                    self.OUT_P_d[k][self.t_act] = self.P_d[k,self.Time_Stamp-1].x
                    self.cum_sum += round(self.d_i[k,self.Time_Stamp-1].x)*self.ST[k]['AD'] + self.P_d[k,self.Time_Stamp-1].x*self.ST[k]['BD'] + round(self.c_i[k,self.Time_Stamp-1].x)*self.ST[k]['AC'] + self.P_d[k,self.Time_Stamp-1].x*self.ST[k]['BC']
            
            #Grid price
            self.Init_Conditions['P_g'][t-1] = self.P_g[t].x
        
        #Pomocna promenna na ukladani vysledku
        self.t_act +=1
        '''
        Jsme na konci dne
            -> Algoritmus konci, uloz si hodnoty
        '''
        if(self.t_act == 475):
            for t in range(self.Time_Stamp,self.N_t):
                for k in range(self.N_P):
                    # self.OUT_P_i[k][50+t] = self.P_i[k,self.Time_Stamp].x
                    self.OUT_P_i[k][470+t] = self.P_i[k,t].x
                    self.cum_sum+= round(self.y_a[k,t].x)*self.PP[k]['C_A'] + round(self.y_d[k,t].x)*self.PP[k]['C_D'] + round(self.x_i[k,t].x)*self.PP[k]['A'] + self.P_i[k,t].x*self.PP[k]['B']

                for k in range(self.N_S):
                    # self.OUT_P_c[k][50+t] = self.P_c[k,self.Time_Stamp].x
                    # self.OUT_P_d[k][50+t] = self.P_d[k,self.Time_Stamp].x
                    self.OUT_P_c[k][470+t] = self.P_c[k,t].x
                    self.OUT_P_d[k][470+t] = self.P_d[k,t].x
                    self.cum_sum += round(self.d_i[k,t].x)*self.ST[k]['AD'] + self.P_d[k,t].x*self.ST[k]['BD'] + round(self.c_i[k,t].x)*self.ST[k]['AC'] + self.P_d[k,t].x*self.ST[k]['BC']

    def save_init_cond_first(self):
        '''
        Funkce na ulozeni pocatecnich podminek pri prvnim spusteni algoritmu 
            -> algoritmus se inicializuje a nasledne si pamatuje prvnich 5 kroku
        '''
        self.Init_Conditions['P_i'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['x_i'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['y_a'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['y_d'] = [[0]*self.N_P]*self.Time_Stamp
        self.Init_Conditions['SOC_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_c'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_d'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['c_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['d_i'] = [[0]*self.N_S]*self.Time_Stamp
        self.Init_Conditions['P_g'] = [0]*self.Time_Stamp

        for t in range(self.Time_Stamp):
            self.t_act=self.Time_Stamp
            #Power plants
            for k in range(self.N_P):
                self.Init_Conditions['P_i'][t][k] = self.P_i[k,t].x
                self.Init_Conditions['x_i'][t][k] = 1 if (self.x_i[k,t].x > 0.75) else 0
                self.Init_Conditions['y_a'][t][k] = 1 if (self.y_a[k,t].x > 0.75) else 0
                self.Init_Conditions['y_d'][t][k] = 1 if (self.y_d[k,t].x > 0.75) else 0
                self.OUT_P_i[k][t] = self.P_i[k,t].x
                self.cum_sum+= round(self.y_a[k,t].x)*self.PP[k]['C_A'] + round(self.y_d[k,t].x)*self.PP[k]['C_D'] + round(self.x_i[k,t].x)*self.PP[k]['A'] + self.P_i[k,t].x*self.PP[k]['B']
            
            #Storage systems
            for k in range(self.N_S):
                self.Init_Conditions['SOC_i'][t][k] = self.SOC_i[k,t].x
                self.Init_Conditions['P_c'][t][k] = self.P_c[k,t].x
                self.Init_Conditions['P_d'][t][k] = self.P_d[k,t].x
                self.Init_Conditions['c_i'][t][k] = 1 if (self.c_i[k,t].x > 0.75) else 0
                self.Init_Conditions['d_i'][t][k] = 1 if (self.d_i[k,t].x > 0.75) else 0
                self.OUT_P_c[k][t] = self.P_c[k,t].x
                self.OUT_P_d[k][t] = self.P_d[k,t].x
                self.cum_sum += round(self.d_i[k,t].x)*self.ST[k]['AD'] + self.P_d[k,t].x*self.ST[k]['BD'] + round(self.c_i[k,t].x)*self.ST[k]['AC'] + self.P_d[k,t].x*self.ST[k]['BC']

            #Grid price
            self.Init_Conditions['P_g'][t] = self.P_g[t].x

    def print_results(self):
        """
        Funkce uklada data jako .json file
        """
        output_dict = dict()
        PP_array = []
        ST_array = []
        PG_array = []
        for k in range(self.N_P):
            PP_dict = dict()
            P_t = []
            x_t = []
            a_t = []
            d_t = []
            for t in range(self.N_t):
                P_t.append(self.P_i[k,t].x)
                x_t.append(int(self.x_i[k,t].x))
                a_t.append(int(self.y_a[k,t].x))
                d_t.append(int(self.y_d[k,t].x))
            PP_dict['P_t'] = P_t
            PP_dict['x_t'] = x_t
            PP_dict['a_t'] = a_t
            PP_dict['d_t'] = d_t
            PP_array.append(PP_dict)
        
        for k in range(self.N_S):
            ST_dict = dict()
            PC_t = []
            PD_t = []
            SOC_t = []
            c_t = []
            d_t = []
            for t in range(self.N_t):
                PC_t.append(self.P_c[k,t].x)
                PD_t.append(self.P_d[k,t].x)
                SOC_t.append(self.SOC_i[k,t].x)
                c_t.append(int(round(self.c_i[k,t].x)))
                d_t.append(int(round(self.c_i[k,t].x)))
            ST_dict['SOC_t'] = SOC_t
            ST_dict['PD_t'] = PD_t
            ST_dict['PC_t'] = PC_t
            ST_dict['d_t'] = d_t
            ST_dict['c_t'] = c_t
            ST_array.append(ST_dict)

        for t in range(self.N_t):
            PG_array.append(self.P_g[t].x)

        output_dict['PP'] = PP_array
        output_dict['ST'] = ST_array
        output_dict['Grid'] = PG_array
        self.plot_result(PP_array,ST_array,PG_array)
        print(output_dict)
        json_string = json.dumps(output_dict)
        with open('json_data.json', 'w') as outfile:
            json.dump(json_string, outfile)


    def plot_final(self,aFRR,mFRR,PDG,blstm):
        '''
        Funkce vytvori graficke vystupy
        '''
        P_DG = np.array(self.P_DG)
        aFRRp_plot = np.zeros(len(aFRR))
        aFRRm_plot = np.zeros(len(aFRR))
        mFRRp_plot = np.zeros(len(mFRR))
        mFRRm_plot = np.zeros(len(mFRR))

        for k in range(0,len(aFRR)):
             aFRRp_plot[k] = aFRR[k]['A+']
             aFRRm_plot[k] = aFRR[k]['A-']
        for k in range(0,len(aFRR)):
             mFRRp_plot[k] = mFRR[k]['A+']
             mFRRm_plot[k] = mFRR[k]['A-']
        P_t = self.OUT_P_i
        P_c = self.OUT_P_c
        P_d = self.OUT_P_d
        nums = np.array([1,1,1,1])
        nums_sum = np.cumsum(nums)
        Melnik = np.zeros_like(aFRRm_plot)
        Pocerady = np.zeros_like(aFRRm_plot)
        Ledvice = np.zeros_like(aFRRm_plot)
        Orlik = np.zeros_like(aFRRm_plot)
        for k in range(len(P_t)):
            if(k<nums_sum[0]):
                Melnik += P_t[k]
                continue
            if(k<nums_sum[1]):
                Pocerady += P_t[k]
                continue
            if(k<nums_sum[2]):
                Ledvice += P_t[k]
                continue
            if(k<nums_sum[3]):
                Orlik += P_t[k]
                continue
        #Akumulatory
        nums = [3,1]
        nums_sum = np.cumsum(nums)
        print("Cumsum:")
        print(nums_sum)
        Silna_C = np.zeros_like(aFRRm_plot)
        Slaba_1_C = np.zeros_like(aFRRm_plot)
        Slaba_2_C = np.zeros_like(aFRRm_plot)
        Slaba_3_C = np.zeros_like(aFRRm_plot)
        Silna_D = np.zeros_like(aFRRm_plot)
        Slaba_1_D = np.zeros_like(aFRRm_plot)
        Slaba_2_D = np.zeros_like(aFRRm_plot)
        Slaba_3_D = np.zeros_like(aFRRm_plot)
        Dlouhe_Strane_D = np.zeros_like(aFRRm_plot)
        Dalesice_D = np.zeros_like(aFRRm_plot)
        Bateriova_Stanice_D = np.zeros_like(aFRRm_plot)
        for k in range(len(P_c)):
            if(k==0):
                Slaba_1_C += P_c[k]
                Slaba_1_D += P_d[k]
                continue
            if(k==1):
                Slaba_2_C += P_c[k]
                Slaba_2_D += P_d[k]
                continue
            if(k==2):
                Slaba_3_C += P_c[k]
                Slaba_3_D += P_d[k]
                continue
            if(k==4):
                Silna_C += P_c[k]
                Silna_D += P_d[k]
                continue
        

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(Melnik,'r-',label=r"$P^1_t$")
        axs[0, 0].set_xlabel("t [min]")
        axs[0, 0].set_ylabel("P [MW]")
        axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_ylim([np.min(Melnik[5:475])-1,np.max(Melnik[5:475])+1])
        axs[0, 0].set_xlim([5,475])

        axs[1, 0].plot(Pocerady,color='C1',linestyle = "-",label=r"$P^2_t$")
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].set_xlabel("t [min]")
        axs[1, 0].set_ylabel("P [MW]")
        axs[1, 0].legend()
        axs[1, 0].grid()
        axs[1, 0].set_ylim([np.min(Pocerady[5:475])-1,np.max(Pocerady[5:475])+1])
        axs[1, 0].set_xlim([5,475])

        axs[0, 1].plot(Ledvice,'b-',label=r"$P^3_t$")
        axs[0, 1].set_xlabel("t [min]")
        axs[0, 1].set_ylabel("P [MW]")
        axs[0, 1].legend()
        axs[0, 1].grid()
        axs[1, 1].set_ylim([np.min(Ledvice[5:475])-1,np.max(Ledvice[5:475])+1])
        axs[0, 1].set_xlim([5,475])

        axs[1, 1].plot(Orlik,'c-',label=r"$P^4_t$")
        axs[1, 1].set_xlabel("t [min]")
        axs[1, 1].set_ylabel("P [MW]")
        axs[1, 1].legend()
        axs[1, 1].set_ylim([np.min(Orlik[5:475])-1,np.max(Orlik[5:475])+1])
        axs[1, 1].set_xlim([5,475])
        axs[1, 1].grid()
        fig.tight_layout()
        if blstm:
            plt.savefig("./obrazky/Rozlozeni_Limit_ER_LSTM.pdf")
        else:
            plt.savefig("./obrazky/Rozlozeni_Limit_ER_ARIMA.pdf")
        plt.show()

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(Slaba_1_C,color=cnames['green'],linestyle = "-",     label=r"$P^{1C}_t$")
        axs[0, 0].plot(Slaba_1_D,color=cnames['lightgreen'],linestyle = "-",label=r"$P^{1D}_t$")
        axs[0, 0].set_xlabel("t [min]")
        axs[0, 0].set_ylabel("P [MW]")
        axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_xlim([5,475])

        axs[1, 0].plot(Slaba_2_C,color=cnames['blue'],linestyle = "-",      label=r"$P^{2C}_t$")
        axs[1, 0].plot(Slaba_2_D,color=cnames['aqua'],linestyle = "-",      label=r"$P^{2D}_t$")
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].set_xlabel("t [min]")
        axs[1, 0].set_ylabel("P [MW]")
        axs[1, 0].legend()
        axs[1, 0].grid()
        axs[1, 0].set_xlim([5,475])

        axs[0, 1].plot(Slaba_3_C,color=cnames['red'],linestyle = "-",       label=r"$P^{3C}_t$")
        axs[0, 1].plot(Slaba_3_D,color=cnames['pink'],linestyle = "-",      label=r"$P^{3D}_t$")
        axs[0, 1].set_xlabel("t [min]")
        axs[0, 1].set_ylabel("P [MW]")
        axs[0, 1].legend()
        axs[0, 1].grid()
        axs[0, 1].set_xlim([5,475])

        axs[1, 1].plot(Silna_C,color=cnames['orange'],linestyle = "-",      label=r"$P^{4C}_t$")
        axs[1, 1].plot(Silna_D,color=cnames['yellow'],linestyle = "-",      label=r"$P^{4D}_t$")
        axs[1, 1].set_xlabel("t [min]")
        axs[1, 1].set_ylabel("P [MW]")
        axs[1, 1].legend()
        axs[1, 1].set_xlim([5,475])
        axs[1, 1].grid()
        fig.tight_layout()
        if blstm:
            plt.savefig("./obrazky/Rozlozeni_Limit_ESS_LSTM.pdf")
        else:
            plt.savefig("./obrazky/Rozlozeni_Limit_ESS_ARIMA.pdf")
        plt.show()

def predict_value(signal, time, coeficients,theta):
    '''
    Pomocna funkce pro autoregresni iteraci v pripade predikce pomoci ARIMA modelu
    '''
    y_hat = signal[time]
    for k in range(len(coeficients)):
        y_hat+= -1*coeficients[k]*(signal[time-k]-signal[time-k-1])
    y_hat += theta
    return y_hat

def predict_value_recurent(signal, time, coeficients,theta, predicted):
    '''
    Pomocna funkce pro autoregresni iteraci v pripade predikce pomoci ARIMA modelu
    '''
    y_hat = predicted[0]
    for k in range(len(coeficients)):
        if(k<len(predicted)-1):
            y_hat+= -1*coeficients[k]*(predicted[k]-predicted[k+1])
        elif(k==len(predicted)-1):
            y_hat+= -1*coeficients[k]*(predicted[k]-signal[time-k-1])
        else:
            y_hat+= -1*coeficients[k]*(signal[time-k]-signal[time-k-1])
    y_hat += theta
    return y_hat

def get_imputs_ARIMA(aFRR_demand, aFRR_init, WindowSize,coeficients, Theta):
    '''
    Funkce provede predikci hodnot pomoci ARIMA modelu
    '''
    aGen = aFRR_Generator(WindowSize)
    len_demand = np.size(aFRR_demand)-1
    aFRR_step1 = np.copy(aFRR_demand)
    aFRR_step2 = np.copy(aFRR_demand)
    aFRR_step3 = np.copy(aFRR_demand)
    aFRR_step4 = np.copy(aFRR_demand)
    aFRR_pred = np.zeros(4)
    aFRR_pred[0] = predict_value(signal=aFRR_step1,time=len_demand,coeficients=coeficients,theta=Theta)
    aFRR_step2 = np.hstack((aFRR_step2[1:],aFRR_pred[0:1]))
    aFRR_pred[1] = predict_value(signal=aFRR_step2,time=len_demand,coeficients=coeficients,theta=Theta)
    aFRR_step3 = np.hstack((aFRR_step3[2:],aFRR_pred[0:2]))
    aFRR_pred[2] = predict_value(signal=aFRR_step3,time=len_demand,coeficients=coeficients,theta=Theta)
    aFRR_step4 = np.hstack((aFRR_step4[3:],aFRR_pred[0:3]))
    aFRR_pred[3] = predict_value(signal=aFRR_step4,time=len_demand,coeficients=coeficients,theta=Theta)
    aFRR_demand = np.hstack((aFRR_demand,aFRR_pred))
    aFRR_input = np.copy(aFRR_demand[-WindowSize:])
    aGen.aFRR_Demand = aFRR_input
    aGen.create_response(aFRR_init)
    aGen.create_interval()
    aFRR_ic = aGen.aFRR_Response[1]
    
    return aGen.aFRRp, aGen.aFRRm, aFRR_ic

def get_inputs_LSTM(aFRR_demand, aFRR_init, WindowSize, Predictor : LSTM_Predictor):
    '''
    Funkce provede predikci hodnot pomoci LSTM modelu
    '''
    aGen = aFRR_Generator(WindowSize)
    tmp_demand = np.copy(aFRR_demand[-15:])
    Predictor.load_TS(tmp_demand)
    aFRR_pred = Predictor.predict_aFRR()
    aFRR_input = np.hstack((aFRR_demand[-5:],aFRR_pred))

    aGen.aFRR_Demand = aFRR_input
    aGen.create_response(aFRR_init)
    aGen.create_interval()
    aFRR_ic = aGen.aFRR_Response[1]

    return aGen.aFRRp, aGen.aFRRm, aFRR_ic

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print(b)
    print(a)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main():
    Time_Stamp = 4
    Window_Size = 2*Time_Stamp+1
    
    #ARIMA init
    with open('ARIMA_coef.npy', 'rb') as f:
        coeficients = np.load(f)
    Theta = 1 + np.sum(coeficients)
    
    #LSTM init
    Predictor = LSTM_Predictor()

    #Load Data
    tst_data = dict()
    tst_data['PP'] = gen.add_PP()
    tst_data['ST'] = gen.add_ST()
    tst_data['P_DG'] = gen.add_PDG()
    tst_data['mFRR'], mFRR_demand = gen.add_mFRR()
    tst_data['aFRR'] = gen.add_aFRR()
    tst_data['Grid'] = gen.add_Grid()
    
    #Create model
    UUT = My_Model()
    #Init model
    Params_P_DG = tst_data['P_DG'][0:(Time_Stamp+1)]
    Params_aFRR = tst_data['aFRR'][0:(Time_Stamp+1)]
    Params_mFRR = tst_data['mFRR'][0:(Time_Stamp+1)]
    Params_Grig = tst_data['Grid'][0:(Time_Stamp+1)]
    UUT.load_params(tst_data['PP'],tst_data['ST'],Params_P_DG,Params_aFRR,Params_mFRR,Params_Grig,0)
    UUT.create_variables()
    UUT.add_constrains()
    UUT.set_objective()
    UUT.calculate()
    UUT.change_Time(4)
    UUT.save_init_cond_first()
    k = 0
    with open("Test_Signal.npy",'rb') as f:
        aFRR_tst = np.load(f)
        aFRR_tst = aFRR_tst.flatten()
    aFRR_tst1 = np.copy(aFRR_tst)
    aFRR_filter = butter_lowpass_filter(aFRR_tst1,1.5,10,1)
    aFRR_init = 0

    #Use LSTM predictor or ARIMA predictor
    blstm = True

    #Plot results
    bplot = False
    #Real-time simulation
    while((k+9)<len(tst_data['P_DG'])):
        Params_P_DG = deepcopy(tst_data['P_DG'][k:(k+9)])
        Params_aFRR = deepcopy(tst_data['aFRR'][k:(k+9)])
        Params_mFRR = deepcopy(tst_data['mFRR'][k:(k+9)])
        Params_Grig = deepcopy(tst_data['Grid'][k:(k+9)])
        if blstm:
            aFRRp, aFRRm, aFRR_init = get_inputs_LSTM(aFRR_demand=deepcopy(aFRR_filter[:20+k]), aFRR_init=aFRR_init, WindowSize=9, Predictor=Predictor)
        else:
            aFRRp, aFRRm, aFRR_init = get_imputs_ARIMA(aFRR_demand=aFRR_filter[:20+k], aFRR_init=aFRR_init, WindowSize=9, coeficients=coeficients, Theta=Theta)
        for l in range(9):
            Params_aFRR[l]['A+'] = aFRRp[l]
            Params_aFRR[l]['A-'] = aFRRm[l]
            if(5+k+l<480):
                tst_data['aFRR'][5+k+l]['A+'] = aFRRp[l]
                tst_data['aFRR'][5+k+l]['A-'] = aFRRm[l]
        UUT.load_params(tst_data['PP'],tst_data['ST'],Params_P_DG,Params_aFRR,Params_mFRR,Params_Grig,4)
        UUT.create_variables()
        UUT.add_constrains()
        UUT.set_objective()
        UUT.calculate()
        UUT.save_init_cond()
        k+=1
    if bplot:
        t = np.linspace(1,480,480)
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(t,aFRR_filter[10:490],linewidth=1.5,color='blue',label=r"$aFRR^a_t$")
        ax1.grid()
        ax1.legend()
        ax1.set_xlim([5,475])
        ax1.set_ylabel("P [MW]")
        ax2.plot(t,mFRR_demand,linewidth=1.5,color='red',label=r"$mFRR^a_t$")
        ax2.set_xlim([5,475])
        ax2.set_ylabel("P [MW]")
        ax2.set_xlabel("t [min]")
        ax2.grid()
        ax2.legend()
        plt.savefig('./obrazky/Pozadavky_Limit_AB_CEZ.pdf')
        plt.show()
        UUT.plot_final(tst_data['aFRR'],tst_data['mFRR'],tst_data['P_DG'],blstm)
    print("Total cost: ",UUT.cum_sum)

if __name__ == "__main__":
    main()