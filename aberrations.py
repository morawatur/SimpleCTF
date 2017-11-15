import numpy as np

# ---------------------------------------------------------------

def polar2complex(amp, phs):
    return amp * np.exp(1j * phs)

# ---------------------------------------------------------------

def complex2polar(x):
    return np.abs(x), np.angle(x)

# ---------------------------------------------------------------

class Astigmatism:
    def __init__(self, am, ph):
        self.amp = am
        self.phs = ph
        self.A1 = polar2complex(am, ph)

    def real(self):
        return self.A1.real

    def imag(self):
        return self.A1.imag

    def set_re_im(self, re, im):
        self.A1 = np.complex64(re, im)
        self.amp, self.phs = complex2polar(self.A1)

    def set_am_ph(self, am, ph):
        self.amp = am
        self.phs = ph
        self.A1 = polar2complex(am, ph)

# ---------------------------------------------------------------

def calc_phi_A1(A1):
    phi_A1 = 0+0j
    # phi_A1.real = ...
    # phi_A1.imag = ...
    return

# ---------------------------------------------------------------

def calc_phi_A2(A2):
    pass