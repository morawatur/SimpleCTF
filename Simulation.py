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

    ctf_amp, ctf_phase = ctf_calc.calc_ctf_2d(img_dim, px_dim, ctf_calc.ewf_length, defocus, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)
    # ctf_calc.save_image(ctf_amp, 'ctf_amp.png', np.min(ctf_amp), np.max(ctf_amp))
    # ctf_calc.save_image(ctf_phase, 'ctf_phase.png', np.min(ctf_phase), np.max(ctf_phase))

    sim_fft_amp = img_fft_amp * ctf_amp
    sim_fft_phase = img_fft_phase + ctf_phase

    sim_fft = polar2complex(sim_fft_amp, sim_fft_phase)
    sim = np.fft.ifft2(sim_fft)

    sim_amp, sim_phase = complex2polar(sim)
    return sim_amp, sim_phase

# ---------------------------------------------------------------

img_data, px_dims = dm3.ReadDm3File('lat5.dm3')
px_sz = 40e-12

amplitudes = []
phases = []

am_lims = [1e9, 0]
ph_lims = [0, 0]

df_values1 = np.arange(0.0, 105e-9, 10e-9)
df_values2 = np.arange(120e-9, 210e-9, 20e-9)
df_values3 = np.arange(250e-9, 1040e-9, 50e-9)
df_values = np.concatenate((df_values1, df_values2, df_values3))

for df in df_values:
    amplitude, phase = simulate_image_for_defocus(img_data, px_sz, df)
    amplitudes.append(amplitude)
    phases.append(phase)
    am_min, am_max = np.min(amplitude), np.max(amplitude)
    ph_min, ph_max = np.min(phase), np.max(phase)
    if am_min < am_lims[0]: am_lims[0] = am_min
    if am_max > am_lims[1]: am_lims[1] = am_max
    if ph_min < ph_lims[0]: ph_lims[0] = ph_min
    if ph_max > ph_lims[1]: ph_lims[1] = ph_max

for df, idx in zip(df_values, range(df_values.shape[0])):
    am_fn = 'df_sim_Cs/am_{0:.0f}nm.png'.format(df * 1e9)
    ph_fn = 'df_sim_Cs/ph_{0:.0f}nm.png'.format(df * 1e9)
    ctf_calc.save_image(amplitudes[idx], am_fn, am_lims[0], am_lims[1])
    ctf_calc.save_image(phases[idx], ph_fn, ph_lims[0], ph_lims[1])
    # img_sim = simulate_image_for_defocus_PyEWRec(img_data, px_sz, df)
    # imsup.SaveAmpImage(img_sim, am_fn)
    # imsup.SavePhaseImage(img_sim, ph_fn)


