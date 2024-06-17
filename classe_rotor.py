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

    def __init__(self, mu, d):
        self.mu = mu
        self.d = d


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
        G = np.array([[0,(-self.a())], [(self.a()),0]])
        return G

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
                    la.solve(-self.M(), (self.G()*rot_rpm))])])
        return A


    def campbell(self, rot_max, order):
        rot_rpm1 = (np.linspace(0, rot_max, 200)) # Criando o vetor de velocidades em rpm
        omega=rot_rpm1*2*pi/60   # Criando o vetor de velocidades em rad/s
        wn_fw = []               #abrindo uma lista para receber os valores das frequencias naturais para cada rotação
        wn_bw = []
        for w in omega:
            AUTOVAL= la.eig(self.A(w))[0]     # Iterando para cada valor de w in omega em rad/s
            wn = np.imag(AUTOVAL)
            wn_fw.append(wn[0])               # Pegando os valores positivos referentes a coluna 0
            wn_bw.append(wn[2])               # Pegando os valores positivos referentes a coluna 2
        wn_fw = np.array(wn_fw)/(2*pi)         #Transformando de lsita para array para efetuar a transf de rad/s para Hz
        wn_bw = np.array(wn_bw)/(2*pi)
        plt.plot(rot_rpm1, wn_fw, c='r', linestyle='-')
        plt.plot(rot_rpm1, wn_bw, c='b', linestyle='-')
        plt.plot(rot_rpm1, rot_rpm1 / (order * 60), c='black', linestyle='-')
        plt.title('Campbell Diagram')
        plt.xlabel('Speed (rpm)')
        plt.ylabel('Frequency (Hz)')
        plt.legend(("FW", "BW", f"{order}x"), loc="upper left")
        plt.axis([0, rot_max, 0, 80])
        plt.grid(True)
        return plt.show()

#         print(a)


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
unb = Unbalance(mu = 10**-4, d = 0.15)
bearing = Bearing(1.195e+6/2, 1.195e+6/2,1.195e+6/2, 1.195e+6/2, (0.8/3))

rotor_assembled = Rotor(eixo, disco1, unb, bearing)
rotor_assembled.campbell(9000,1)

