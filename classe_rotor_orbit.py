import numpy as np
import math
#import ccp
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import linalg as la
from scipy import signal as signal
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton
from scipy.sparse import linalg as las
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
#Q_ = ccp.Q_

pi = math.pi


class Shaft:

    def __init__(self, L, R_out, R_in, rho, E):
        self.L = L          # comprimento do Eixo
        self.R_out = R_out  # raio externo em metros
        self.R_in = R_in    # raio interno em metros
        self.rho = rho      #  massa volumétrica  kg/m^3
        self.E = E          # módulo de Young N/m^2

        self.S = pi*(self.R_out**2)     #Área da seção transversal do eixo m^2
        self.I = 0.25*pi*(self.R_out**4)  #momento de inércia para o eixo Y (longitudinal)
        self.Ms = pi * (self.R_out ** 2)*(self.rho*self.L)   # massa kg do eixo


class Disc:

    def __init__(self, r_out, r_in, d, rho, l1):
        self.r_out = r_out
        self.r_in = r_in
        self.d = d        # Espessura do disco
        self.rho = rho    #  massa volumétrica  kg/m^3
        self.l1 = l1      # Posição do disco ao longo do eixo 0<l1>L

        self.Md = pi*((self.r_out**2)-(self.r_in**2))*d*rho                      # Massa kg do disco
        self.Idx = (self.Md*((3*self.r_out**2)+(3*self.r_in**2)+self.d**2))/12   # Momento de Inércia em X e Z são iguais Eq. 27 pág. 13 LaLane
        self.Idy = self.Md*0.5*((self.r_out**2)-(self.r_in**2))                  # Momento de Inércia em Y (Longitudinal)


class Unbalance:

    def __init__(self, mu, d, l_unb):
        self.mu = mu #massa de desnbalanceamento em kg
        self.d = d #posição da massa, em geral raio externo do disco
        self.l_unb = l_unb #distância da massa em relação à posição 0, em metros


class Bearing:

    def __init__(self, kxx1, kxx2, kzz1, kzz2, l2):
        self.kxx1 = kxx1   # Rigidez na direção X do mancal 1
        self.kxx2 = kxx2   # Rigidez na direção X do mancal 2
        self.kzz1 = kzz1   # Rigidez na direção Z do mancal 1
        self.kzz2 = kzz2   # Rigidez na direção Z do mancal 2
        self.l2 = l2     # Posição do disco ao longo do eixo 0<l2>L

# Editar parte do rotor de modo a ser possível criar um rotor com dois ou mais discos, criando lista de discos


class Rotor:
    def __init__(self, shaft, disc, unbalance, bearing):
        self.shaft = shaft
        self.disc = disc
        self.unbalance = unbalance
        self.bearing = bearing

    def M(self):
        M = np.array([[self.m(), 0], [0, self.m()]])
        return M

    def m(self): # Criando a 'matriz massa' 2x2 com a definição do Lalane na pág.12 equação 12 e 13, o valor do parâmetro 'm' é igual a
        m = (self.disc.Md * f(self.disc.l1, self.shaft.L) ** 2) + (
             self.disc.Idx * g(self.disc.l1, self.shaft.L) ** 2) + (
             self.shaft.rho * self.shaft.S * f_int_square(0, self.shaft.L)) + (
             self.shaft.rho * self.shaft.I * g_int_square(0, self.shaft.L))
        return m

    def K(self):  #criando a matriz de rigidez 2x2 com os valores de kzz e kxx para 2 mancais distintos
        K = np.array([[(self.bearing.kxx1+self.bearing.kxx2), 0], [0, (self.bearing.kzz1+self.bearing.kzz2)]])
        return K


    def G(self):  # criando a matriz giroscópica 2x2 com velocidade 'omega' rad/s e parâmetro 'a'
        G = np.array([[0,(-self.a())], [(self.a()),0]])    # inserindo um alto amortecimento arbitrário para que o integrador numérico não fique instável
        return G

    def C(self):  # criando a matriz amortecimento, usando o AMORTECiMENTO PROPORCiONAL conforme equação 2.259 e 2.260 da pág. 50
        beta = 0.0002
        C = np.array([[beta*1.5e+05,0], [0,beta*3.75e+05]])    # inserindo um alto amortecimento arbitrário para que o integrador numérico não fique instável
        return C

    # Pegando a definição do Lalane na pág.12 equação 12 e 13, o valor do parâmetro 'a' é igual a Idy*g(l1)**2+2rho*I*(integral de g(L))
    def a(self):
        a = (self.disc.Idy * g(self.disc.l1, self.shaft.L) ** 2) + (
                2 * self.shaft.rho * self.shaft.I * g_int_square(0, self.shaft.L))
        return a


# DEFININDO a MATRIZ A  """State space matrix for an instance of a rotor.

    def A(self, rot_rpm):
        Z = np.zeros((2,2))
        I = np.eye(2)

        A = np.vstack(
                    [np.hstack([Z, I]),
                    np.hstack([la.solve(-self.M(), self.K()),
                    la.solve(-self.M(), self.C()+self.G()*rot_rpm)])])
        return A


        # Calculando as fequencias naturais através da matriz dos autovalores da matriz A ---> STATE SPACE MATRX

    def campbell(self, rot_max):
        rot_rpm1 = (np.linspace(0, rot_max, 2000)) # Criando o vetor de velocidades em rpm
        omega=rot_rpm1*2*pi/60   # Criando o vetor de velocidades em rad/s
        wn_fw = []               #abrindo uma lista para receber os valores das frequencias naturais para cada rotação
        wn_bw = []
        for w in omega:
            AUTOVAL= la.eig(self.A(w))[0]     # Iterando para cada valor de w in omega em rad/s
            wn = np.imag(AUTOVAL)
            wn_fw.append(np.max(np.abs(wn)))   # Pegando os valores máximos que correspondem as fequencias forward
            wn_bw.append(np.min(np.abs(wn)))   # Pegando os valores máximos que correspondem as fequencias forward
        wn_fw = np.array(wn_fw)/(2*pi)         #Transformando de lsita para array para efetuar a transf de rad/s para Hz
        wn_bw = np.array(wn_bw)/(2*pi)
        plt.plot(rot_rpm1, wn_fw, c='r', linestyle='-')
        plt.plot(rot_rpm1, wn_bw, c='b', linestyle='-')
        plt.plot(rot_rpm1, rot_rpm1 / (60), c='black', linestyle='-')
        plt.plot(rot_rpm1, rot_rpm1 / (2*60), c='grey', linestyle='-')
        plt.title('Campbell Diagram')
        plt.xlabel('Speed (rpm)')
        plt.ylabel('Frequency (Hz)')
        plt.legend(("FW", "BW", f"1x",f"1/2x"), loc="upper left")
        plt.axis([0, rot_max, 0, 80])
        plt.grid(True)
        return plt.show()

    # calculando agr numericamente através do ODEiNT as respostas do sistema a uma força F


    # Agora temos a matriz B no Space state matrix pois temos uma força F atuando no sistema

    def B_state_space(self):
        B_state_space = np.block([
        [np.zeros((2, 2))],
        [la.inv(self.M())]
        ])
        return B_state_space

# Amplitude da Força de Desbalanceamento

    def c_unb(self):
        c_unb = self.unbalance.mu * self.unbalance.d * f(self.unbalance.l_unb, self.shaft.L)
        return c_unb

    # Initial conditions
    def z0(self):
        return np.array([0, 0, 0, 0])

        # Define the synchoronous force vector function due to a unbalance mass mu
    def force_sync(self,t,omega):
        F1 = self.c_unb() * (omega**2)
        F2 = self.c_unb() * (omega ** 2)
        s = 1                      # para quando a freq. de excitação for um múltiplo da rotação (Assíncrona)
        force_sync = np.array([F1 * np.sin(s * omega * t), F2 * np.cos(s * omega * t)])
        return force_sync

    # Define the Assynchoronous force vector function
    def force_assync(self,t,omega):
        F1 = 1
        F2 = 1
        s = 0.5       # para quando a freq. de excitação for um múltiplo da rotação (Assíncrona)
        force_assync = np.array([F1 * np.sin(s * omega * t), F2 * np.cos(s * omega * t)])
        return force_assync

    # Define the differential equation dxdt = A @ Z + B @ force
    def state_space_model(self,t,z, omega):
        dzdt = self.A(omega) @ z + self.B_state_space() @ self.force_sync(t,omega)
        return dzdt


    # Time span for numerical integration
    def t_span(self,omega):
        t_span = (0, (300*pi / omega))
        return t_span

    # Time vector for numerical integration
    def t_eval(self, omega):
        t_eval = np.linspace(1, (300 * pi / omega) , 6000)
        return t_eval

    # Time vector for numerical integration for bode plot in order to minimize tie
    def t_eval_cheap(self, omega):
        t_eval_cheap = np.linspace(1, (300 * pi / omega) , 600)
        return t_eval_cheap


    def amplitude(self,omega):
        solution = solve_ivp(self.state_space_model, self.t_span(omega), self.z0(), t_eval = self.t_eval_cheap(omega),args=[omega])
        # Extract displacements
        q1 = solution.y[0,530:]
        q2 = solution.y[1,530:]
        #t = solution.t[49470:49900]
        # Calculando o módulo das amplitudes em x e z
        Q1 = np.abs(max(q1))#(max(q1) - min(q1))/2
        Q2 = np.abs(max(q2))#(max(q2) - min(q2))/2
        return Q1,Q2

    def bode(self, rot_max):
        rot_rpm1 = (np.linspace(1200, rot_max, 600))  # Criando o vetor de velocidades em rpm
        omega = rot_rpm1 * 2 * pi / 60  # Criando o vetor de velocidades em rad/s
        list_Q1 = []  # abrindo uma lista para receber os valores das frequencias naturais para cada rotação
        list_Q2 = []
        for w in omega:
            amp1, amp2 = self.amplitude(w)
            list_Q1.append(amp1)
            list_Q2.append(amp2)

        plt.figure(figsize=(10, 6))
        plt.plot(rot_rpm1, list_Q1, label='AMP Q1')
        plt.plot(rot_rpm1, list_Q2, label='AMP Q2')
        plt.yscale('log')
        plt.xlabel('Rotação')
        plt.ylabel('Amplitude Q')
        plt.title('AMPLiTUDE pela ROtação ')
        plt.legend()
        return plt.show()

    def orbit(self,omega):
        omega=omega*2*pi/60
        solution = solve_ivp(self.state_space_model, self.t_span(omega), self.z0(), t_eval = self.t_eval(omega),args=[omega])
        # Extract displacements and velocities
        # Extract results
        x1 = solution.y[0,5600:]
        x2 = solution.y[1,5600:]
        t = solution.t[5600:]
        # x1 = solution.y[0, :]
        # x2 = solution.y[1, :]
        # t = solution.t[:]

        # Determinar os limites dos eixos
        x_min = min(min(x1), min(x2))
        x_max = max(max(x1), max(x2))

        # Plot the orbits
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(t, x1, label='x1')
        plt.plot(t, x2, label='x2')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.legend()
        plt.title('Displacement vs Time')

        plt.subplot(1, 2, 2)
        plt.plot(x1, x2)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Orbit')
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.tight_layout()
        return plt.show()

        # The Force Fixed in space vector function

    def force_fix(self, t, omeguinha):
        F1 = 1
        F2 = 0
        force_fix = np.array([F1 * np.sin(omeguinha * t), F2 * np.cos(omeguinha * t)])
        return force_fix
        # Define the differential equation dxdt = A @ Z + B @ force fixed in space

    def state_space_model_force_fix(self, t, z, omega, omeguinha):
        dzdt = self.A(omega) @ z + self.B_state_space() @ self.force_fix(t, omeguinha)
        return dzdt
    def amplitude_force_fix(self,omega, omeguinha):
        solution = solve_ivp(self.state_space_model_force_fix, self.t_span(omeguinha), self.z0(), t_eval = self.t_eval_cheap(omeguinha),args=[omega, omeguinha])
        # Extract displacements
        q1 = solution.y[0,530:]
        q2 = solution.y[1,530:]
        #t = solution.t[49470:49900]
        # Calculando o módulo das amplitudes em x e z
        Q1 = np.abs(max(q1))#(max(q1) - min(q1))/2
        Q2 = np.abs(max(q2))#(max(q2) - min(q2))/2
        return Q1,Q2

    # calculando agora o diagrama de Bode variando a rotação de 0-9000 rpm

    def bode_force_fix(self, omega, rot_max):
        rot_rpm1 = (np.linspace(1200, rot_max, 600))  # Criando o vetor de velocidades em rpm
        omeguinha = rot_rpm1 * 2 * pi / 60  # Criando o vetor de velocidades em rad/s
        list_Q1 = []  # abrindo uma lista para receber os valores das frequencias naturais para cada rotação
        list_Q2 = []
        for w in omeguinha:
            amp1, amp2 = self.amplitude_force_fix(omega,w)
            list_Q1.append(amp1)
            list_Q2.append(amp2)

        plt.figure(figsize=(10, 6))
        plt.plot(rot_rpm1/60, list_Q1, label='AMP Q1')
        plt.plot(rot_rpm1/60, list_Q2, label='AMP Q2')
        plt.yscale('log')
        plt.xlabel('Rotação')
        plt.ylabel('Amplitude Q')
        plt.title('AMPLiTUDE pela ROtação ')
        plt.legend()
        return plt.show()

    def orbit_force_fix(self,omega, omeguinha):
        omega = omega * 2 * pi / 60
        omeguinha = omeguinha * 2 * pi / 60
        solution = solve_ivp(self.state_space_model_force_fix, self.t_span(omeguinha), self.z0(), t_eval = self.t_eval(omeguinha),args=[omega,omeguinha])
        # Extract displacements and velocities
        # Extract results
        x1 = solution.y[0,4500:]
        x2 = solution.y[1,4500:]
        t = solution.t[4500:]
        # x1 = solution.y[0, :]
        # x2 = solution.y[1, :]
        # t = solution.t[:]

        # Determinar os limites dos eixos
        x_min = min(min(x1), min(x2))
        x_max = max(max(x1), max(x2))

        # Plot the orbits
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(t, x1, label='x1')
        plt.plot(t, x2, label='x2')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.legend()
        plt.title('Displacement vs Time')

        plt.subplot(1, 2, 2)
        plt.plot(x1, x2)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Orbit')
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.tight_layout()
        return plt.show()



def g(y, L):
    x = pi * y / L
    g = (pi / L) * math.cos(x)
    return g


def g_int_square(a, b):
    g = lambda x: ((pi / b) * (math.cos(pi * x / b))) ** 2
    g_int_square = integrate.quad(g, a, b)
    return g_int_square[0]

def f(y, L):
    x = pi*y/L
    f = math.sin(x)
    return f


def f_int_square(a, b):
    f = lambda x: (math.sin(pi*x/b))**2
    f_int_square = integrate.quad(f, a, b)
    return f_int_square[0]

def h(y, L):
    h = -((pi/L)**2)*f(y, L)
    return h


def h_int_square(a, b):
    h = lambda x: (-((pi/b)**2)*(math.sin(pi*x/b)))**2
    h_int_square = integrate.quad(h, a, b)
    return h_int_square[0]


# Montando o Eixo


eixo = Shaft(0.4, 0.01, 0,7800, 2e+11)
disco1 = Disc(r_out=0.15, r_in=0.01, d=0.03, rho=7800, l1=0.4/3)
unb = Unbalance(mu = 1e-4, d = 0.15, l_unb=0.4/3)
bearing = Bearing(1.195e+6/2, 1.195e+6/2,1.570e+6/2, 1.570e+6/2, (0.8/3))

rotor_assembled = Rotor(eixo, disco1, unb, bearing)

#print(rotor_assembled.c_unb())
#A=rotor_assembled.A(2520*2*pi/60)
#print(A)
#rotor_assembled.campbell(12000)
#print(rotor_assembled.B_state_space() @ rotor_assembled.force(1))
#rotor_assembled.orbit(2520)
#print(rotor_assembled.amplitude(4000* (2 * pi / 60)))
rotor_assembled.orbit_force_fix(4000,52.75*2*pi/60)
