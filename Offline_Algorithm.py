from re import A
import gurobipy as g
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import testData as gen
from mycolors import cnames

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
    
    def load_params(self,PP_params,ST_params,P_DG,aFRR,mFRR,Grid_price):
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
        self.N_t = len(P_DG)
        self.N_P = len(PP_params)
        self.N_S = len(ST_params)

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
        for t in range(self.N_t):
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
        C_PA = g.quicksum(self.y_a[k,t]*self.PP[k]['C_A'] for k in range(self.N_P) for t in range(self.N_t))
        C_PD = g.quicksum(self.y_d[k,t]*self.PP[k]['C_D'] for k in range(self.N_P) for t in range(self.N_t))
        C_PC = g.quicksum(self.x_i[k,t]*self.PP[k]['A'] + self.P_i[k,t]*self.PP[k]['B'] for k in range(self.N_P) for t in range(self.N_t))
        C_SD = g.quicksum(self.d_i[k,t]*self.ST[k]['AD'] + self.P_d[k,t]*self.ST[k]['BD'] for k in range(self.N_S) for t in range(self.N_t))
        C_SC = g.quicksum(self.c_i[k,t]*self.ST[k]['AC'] + self.P_d[k,t]*self.ST[k]['BC'] for k in range(self.N_S) for t in range(self.N_t))
        C_G = g.quicksum(self.P_g[t]*self.Grid_price[t] for t in range(self.N_t))

        self.VPP_Model.setObjective(C_PA + C_PD + C_PC + C_SD + C_SC + C_G,sense = g.GRB.MINIMIZE)

    def calculate(self):
        '''
        Funkce spousti ILP branch and bound optimalizaci
        '''
        self.VPP_Model.optimize()

    def print_results(self):
        """
        Funkce uklada data jako .json file
        """
        output_dict = dict()
        PP_array = []
        ST_array = []
        PG = []
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
                c_t.append(int(self.c_i[k,t].x))
                d_t.append(int(self.c_i[k,t].x))
            ST_dict['SOC_t'] = SOC_t
            ST_dict['PD_t'] = PD_t
            ST_dict['PC_t'] = PC_t
            ST_dict['d_t'] = d_t
            ST_dict['c_t'] = c_t
            ST_array.append(ST_dict)
        
        for t in range(self.N_t):
            PG.append(self.P_g[t].x)
        
        output_dict['PP'] = PP_array
        output_dict['ST'] = ST_array
        self.plot_result(PP_array,ST_array, PG)
        json_string = json.dumps(output_dict)
        with open('json_data.json', 'w') as outfile:
            json.dump(json_string, outfile)

    def plot_result(self, PP_array, ST_array, grid):
        '''
        Funkce vytvori graficke vystupy
        '''
        P_DG = np.array(self.P_DG)
        aFRRp = np.array(self.aFRR[0]['A+'])
        aFRRm = np.array(self.aFRR[0]['A-'])
        mFRRp = np.array(self.mFRR[0]['A+'])
        mFRRm = np.array(self.mFRR[0]['A-'])
        P_t = np.array(PP_array[0]['P_t'])
        P_c = np.array(ST_array[0]['PC_t'])
        P_d = np.array(ST_array[0]['PD_t'])
        SOC_t = np.array(ST_array[0]['SOC_t'])
        t = np.arange(0, len(P_t), 1)

        nums = [1,1,1,1]
        nums_sum = np.cumsum(nums)
        Melnik = np.zeros_like(aFRRm)
        Pocerady = np.zeros_like(aFRRm)
        Ledvice = np.zeros_like(aFRRm)
        Orlik = np.zeros_like(aFRRm)
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
        Silna_C = np.zeros_like(aFRRm)
        Slaba_1_C = np.zeros_like(aFRRm)
        Slaba_2_C = np.zeros_like(aFRRm)
        Slaba_3_C = np.zeros_like(aFRRm)
        Silna_D = np.zeros_like(aFRRm)
        Slaba_1_D = np.zeros_like(aFRRm)
        Slaba_2_D = np.zeros_like(aFRRm)
        Slaba_3_D = np.zeros_like(aFRRm)
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
        axs[0, 1].set_ylim([np.min(Ledvice[5:475])-1,np.max(Ledvice[5:475])+1])
        axs[0, 1].set_xlim([5,475])

        axs[1, 1].plot(Orlik,'c-',label=r"$P^4_t$")
        axs[1, 1].set_xlabel("t [min]")
        axs[1, 1].set_ylabel("P [MW]")
        axs[1, 1].legend()
        axs[1, 1].set_ylim([np.min(Orlik[5:475])-1,np.max(Orlik[5:475])+1])
        axs[1, 1].set_xlim([5,475])
        axs[1, 1].grid()
        fig.tight_layout()
        plt.savefig("./obrazky/Rozlozeni_Limit_ER_Offline.pdf")
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
        plt.savefig("./obrazky/Rozlozeni_Limit_ESS_Offline.pdf")
        plt.show()

def main():
    bPrint = False
    tst_data = dict()
    tst_data['PP'] = gen.add_PP()
    tst_data['ST'] = gen.add_ST()
    tst_data['P_DG'] = gen.add_PDG()
    tst_data['mFRR'], _ = gen.add_mFRR()
    tst_data['aFRR'], _ = gen.add_aFRR()
    tst_data['Grid'] = gen.add_Grid()
    UUT = My_Model()
    UUT.load_params(tst_data['PP'],tst_data['ST'],tst_data['P_DG'],tst_data['aFRR'],tst_data['mFRR'],tst_data['Grid'])
    UUT.create_variables()
    UUT.add_constrains()
    UUT.set_objective()
    UUT.calculate()
    if bPrint:
        UUT.print_results()

if __name__ == "__main__":
    main()