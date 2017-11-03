import numpy as np

import ctf_calc
import Dm3Reader3 as dm3
import ImageSupport as imsup
import Propagation as prop

# ---------------------------------------------------------------

def polar2complex(amp, phs):
    return amp * np.exp(1j * phs)

# ---------------------------------------------------------------

def complex2polar(x):
    return np.abs(x), np.angle(x)

# ---------------------------------------------------------------

def fft2diff(fft):
    diff = np.zeros(fft.shape, dtype=fft.dtype)
    dim = fft.shape[0]
    for x in range(fft.shape[0]):
        for y in range(fft.shape[1]):
            diff[x, y] = fft[(x + dim // 2) % dim, (y + dim // 2) % dim]
    return diff

# ---------------------------------------------------------------

def simulate_image_for_defocus_PyEWRec(img, px_dim, defocus):
    img1 = imsup.Image(img.shape[0], img.shape[1])
    img1.amPh.am = np.copy(img)
    img1.px_dim = px_dim
    img2 = prop.PropagateBackToDefocus(img1, defocus)
    return img2

# ---------------------------------------------------------------

def simulate_image_for_defocus(img, px_dim, defocus):
    img_dim = img.shape[0]
    img_phase = np.zeros((img_dim, img_dim), dtype=np.float32)
    img_re_im = polar2complex(img, img_phase)
    img_fft = np.fft.fft2(img_re_im)
    img_fft_amp, img_fft_phase = complex2polar(img_fft)
    img_fft_amp = fft2diff(img_fft_amp)
    img_fft_phase = fft2diff(img_fft_phase)
    # ctf_calc.save_image(img_fft_amp, 'fft_amp.png', np.min(img_fft_amp), np.max(img_fft_amp))
    # ctf_calc.save_image(img_fft_phase, 'fft_phase.png', np.min(img_fft_phase), np.max(img_fft_phase))

    # uwzglednic amplitude
    ctf = ctf_calc.calc_ctf_2d(img_dim, px_dim, ctf_calc.ewf_length, defocus, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)
    sim_fft_amp = np.copy(img_fft_amp)
    sim_fft_phase = img_fft_phase + ctf

    sim_fft = polar2complex(sim_fft_amp, sim_fft_phase)
    sim = np.fft.ifft2(sim_fft)

    sim_amp, sim_phase = complex2polar(sim)
    return sim_amp, sim_phase

# ---------------------------------------------------------------

img_data, px_dims = dm3.ReadDm3File('df_sim0.dm3')
px_sz = 80e-12

amplitudes = []
phases = []

am_lims = [1e9, 0]
ph_lims = [1e9, 0]

df_values = np.arange(0.0, 505e-9, 50e-9)

for df, idx in zip(df_values, range(df_values.shape[0])):
    am_fn = 'am_{0:.0f}nm.png'.format(df * 1e9)
    ph_fn = 'ph_{0:.0f}nm.png'.format(df * 1e9)
    # amplitude, phase = simulate_image_for_defocus(img_data, px_sz, df)
    # ctf_calc.save_image(amplitude, am_fn, np.min(amplitude), np.max(amplitude))
    # ctf_calc.save_image(phase, ph_fn, np.min(phase), np.max(phase))
    img_sim = simulate_image_for_defocus_PyEWRec(img_data, px_sz, df)
    imsup.SaveAmpImage(img_sim, am_fn)
    imsup.SavePhaseImage(img_sim, ph_fn)


