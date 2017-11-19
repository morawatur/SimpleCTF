import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
from PIL import ImageFont
from PIL import ImageDraw

import aberrations as ab

ewf_length = 1.97e-12

# ---------------------------------------------------------------

def scale_image(img, old_min, old_max, new_min=0.0, new_max=255.0):
    img_scaled = (img - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return img_scaled

# ---------------------------------------------------------------

def save_image(img, f_path, old_min, old_max, annot=''):
    img_scaled = scale_image(img, old_min, old_max, 0.0, 255.0)
    img_to_save = im.fromarray(img_scaled.astype(np.uint8))

    draw = ImageDraw.Draw(img_to_save)
    fnt = ImageFont.truetype('calibri.ttf', 52)
    draw.text((700, 60), annot, font=fnt, fill='white')

    img_to_save.save(f_path)

# ---------------------------------------------------------------

def save_ctf_plot(ctf_1d, fname='ctf_1d'):
    kx = ctf_1d.kx * 1e-9
    pctf_1d = ctf_1d.get_ctf_sine()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(kx, pctf_1d, 'r', label='PCTF')
    ax.plot(kx, ctf_1d.spat_env, 'g', label='Spatial envelope')
    ax.plot(kx, ctf_1d.temp_env, 'b', label='Temporal envelope')
    legend = ax.legend(loc='lower right', fontsize=12)

    plt.xlim([0, kx[-1]])
    plt.ylim([-1.1, 1.1])
    plt.axhline(0, color='k', lw=1.0)
    plt.xlabel('Spatial frequency k [nm-1]', fontsize=12)
    plt.ylabel('Contrast', fontsize=12)
    # plt.annotate('df = {0:.0f} nm'.format(ctf_1d.abset.C1 * 1e9), xy=(0, 0), xytext=(9.0, 0.9), fontsize=16)

    fig.savefig('{0}.png'.format(fname))
    plt.close(fig)

# ---------------------------------------------------------------

def calc_ctf_1d(img_dim, px_dim, ewf_lambda, defocus, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf1d'):
    df_coeff = np.pi * ewf_lambda * defocus
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs

    rec_px_dim = 1.0 / (img_dim * px_dim)

    x = np.arange(0, img_dim // 2, 1)
    kx = x * rec_px_dim
    k_squared = kx ** 2

    aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)
    pctf = -np.sin(aberr_fun)

    spat_env_fun = np.exp(-((np.pi * conv_angle * kx) ** 2) * (defocus + Cs * ewf_lambda ** 2 * k_squared) ** 2)
    temp_env_fun = np.exp(-(0.5 * np.pi * ewf_lambda * df_spread * k_squared) ** 2)

    pctf *= spat_env_fun
    pctf *= temp_env_fun

    kx *= 1e-9
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(kx, pctf, 'r', label='PCTF')
    ax.plot(kx, spat_env_fun, 'g', label='Spatial coherence envelope')
    ax.plot(kx, temp_env_fun, 'b', label='Temporal coherence envelope')
    legend = ax.legend(loc='lower right', fontsize=14)

    plt.xlim([0, kx[-1]])
    plt.ylim([-1.1, 1.1])
    plt.axhline(0, color='k', lw=1.0)
    plt.xlabel('Spatial frequency k [nm-1]', fontsize=14)
    plt.ylabel('Contrast', fontsize=14)
    plt.annotate('df = {0:.0f} nm'.format(defocus * 1e9), xy=(0, 0), xytext=(9.0, 0.9), fontsize=20)
    plt.annotate('information\nlimit', xy=(9.5, -0.02), xytext=(9.5, -0.4), fontsize=16, horizontalalignment='center',
                 arrowprops=dict(facecolor='black', shrink=0.06))
    # plt.legend()
    fig.savefig('{0}.png'.format(fname))
    plt.close(fig)

    # pctf_to_save = np.vstack((kx, pctf, spat_env_fun, temp_env_fun)).transpose()
    # np.savetxt('pctf1d.txt', pctf_to_save, delimiter='\t')
    print('Done')
    return pctf

# ---------------------------------------------------------------

def calc_ctf_1d_dev(width, bin_dim, aberrations, fname='pctf1d'):
    ctf_1d = ab.ContrastTransferFunction1D(width, bin_dim, aberrations)
    ctf_1d.calc_env_funs()
    ctf_1d.calc_ctf()

    # pctf_1d = ctf_1d.get_ctf_sine()
    save_ctf_plot(ctf_1d, fname)
    return ctf_1d

# ---------------------------------------------------------------

def calc_ctf_2d_dev(img_dim, px_dim, aberrations, fname='pctf2d'):
    ctf_2d = ab.ContrastTransferFunction2D(img_dim, img_dim, px_dim, aberrations)
    ctf_2d.calc_env_funs()
    ctf_2d.calc_ctf()

    # pctf_2d = ctf_2d.get_ctf_sine()
    # save_image(pctf_2d, '{0}.png'.format(fname), -1, 1)
    return ctf_2d

# ---------------------------------------------------------------

def calc_ctf_2d(img_dim, px_dim, ewf_lambda, defocus, A1=0+0j, A2=0.0, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf2d'):
    df_coeff = np.pi * ewf_lambda * defocus
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs
    A11_coeff = np.pi * ewf_lambda * A1.real
    A12_coeff = 2 * np.pi * ewf_lambda * A1.imag
    A2_coeff = (2.0 / 3.0) * np.pi * (ewf_lambda ** 2) * A2

    rec_px_dim = 1.0 / (img_dim * px_dim)
    rec_orig = -1.0 / (2.0 * px_dim)

    x, y = np.mgrid[0:img_dim:1, 0:img_dim:1]
    kx = rec_orig + x * rec_px_dim
    ky = rec_orig + y * rec_px_dim
    k_squared = kx ** 2 + ky ** 2
    k_squares_diff = kx ** 2 - ky ** 2

    aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)
    aberr_fun += A11_coeff * k_squares_diff + A12_coeff * kx * ky       # two-fold astigmatism
    aberr_fun += A2_coeff * kx * (kx ** 2 - 3 * ky ** 2)                # three-fold astigmatism
    pctf = -np.sin(aberr_fun)

    spat_env_fun = np.exp(-(k_squared * (np.pi * conv_angle) ** 2) * (defocus + Cs * ewf_lambda ** 2 * k_squared) ** 2)
    temp_env_fun = np.exp(-(0.5 * np.pi * ewf_lambda * df_spread * k_squared) ** 2)

    pctf *= spat_env_fun
    pctf *= temp_env_fun

    env_funs = spat_env_fun * temp_env_fun

    save_image(pctf, '{0}.png'.format(fname), -1, 1)
    # save_image(pctf, '{0}.png'.format(fname), -1, 1, annot='df = {0:.0f} nm'.format(defocus * 1e9))

    print('Done')
    return env_funs, aberr_fun
    # return pctf

# ---------------------------------------------------------------

def save_range_of_ctf_1d_images(img_dim, px_dim, ewf_lambda, df_pars, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf1d'):

    df_min, df_max, df_step = df_pars
    df_values = np.arange(df_min, df_max, df_step)
    for df in df_values:
        fn = 'pctf1d_new/{0}_{1:.0f}nm'.format(fname, df * 1e9)
        calc_ctf_1d(img_dim, px_dim, ewf_lambda, df, Cs, df_spread, conv_angle, fn)

    print('All done')
    return

# ---------------------------------------------------------------

def save_range_of_ctf_2d_images_dev(img_dim, px_dim, df_pars, aberrations, fname='pctf2d'):
    df_min, df_max, df_step = df_pars
    df0 = aberrations.C1
    df_values = np.arange(df_min, df_max, df_step)

    for df in df_values:
        fn = 'pctf2d_with_A/{0}_{1:.0f}nm'.format(fname, df * 1e9)
        aberrations.set_C1(df)
        calc_ctf_2d_dev(img_dim, px_dim, aberrations, fn)

    aberrations.set_C1(df0)
    print('All done')
    return

# ---------------------------------------------------------------

def save_range_of_ctf_2d_images(img_dim, px_dim, ewf_lambda, df_pars, A_pars, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf2d'):

    df_min, df_max, df_step = df_pars
    df_values = np.arange(df_min, df_max, df_step)

    A_min, A_max, A_step = A_pars
    A_am_values = np.arange(A_min, A_max, A_step)
    nA = (A_max - A_min) // A_step + 1
    A_ph_values = np.arange(0, 2 * np.pi, 2.0 * np.pi / nA)

    for df in df_values:
        for A_am, A_ph, idx in zip(A_am_values, A_ph_values, range(A_am_values.shape[0])):
            fn = 'pctf2d_with_A/{0}_{1:.0f}nm_{2:.0f}'.format(fname, df * 1e9, idx)
            A1 = ab.polar2complex(A_am, A_ph)
            calc_ctf_2d(img_dim, px_dim, ewf_lambda, df, A1, 0.0, Cs, df_spread, conv_angle, fn)

    print('All done')
    return

# ---------------------------------------------------------------

def calc_ctf_2d_PyEWRec(img_dim, px_dim, ewf_lambda, defocus, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf2d'):
    import ImageSupport as imsup

    df_coeff = np.pi * ewf_lambda * defocus
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs

    rec_px_dim = 1.0 / (img_dim * px_dim)
    rec_orig = -1.0 / (2.0 * px_dim)

    x, y = np.mgrid[0:img_dim:1, 0:img_dim:1]
    kx = rec_orig + x * rec_px_dim
    ky = rec_orig + y * rec_px_dim
    k_squared = kx ** 2 + ky ** 2

    aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)
    # pctf = -np.sin(aberr_fun)

    spat_env_fun = np.exp(-(k_squared * (np.pi * conv_angle) ** 2) * (defocus + Cs * ewf_lambda ** 2 * k_squared) ** 2)
    temp_env_fun = np.exp(-(0.5 * np.pi * ewf_lambda * df_spread * k_squared) ** 2)

    aberr_fun *= -1
    env_funs = spat_env_fun * temp_env_fun
    env_funs[env_funs == 0] = 1

    ctf_wf = imsup.Image(img_dim, img_dim, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
    # ctf_wf.amPh.am = np.copy(env_funs)
    ctf_wf.amPh.am = np.ones((img_dim, img_dim), dtype=np.float32)
    ctf_wf.amPh.ph = np.copy(aberr_fun)

    # pctf *= spat_env_fun
    # pctf *= temp_env_fun

    # save_image(pctf, '{0}.png'.format(fname), -1, 1)
    # print('Done')
    return ctf_wf

# ---------------------------------------------------------------

def get_pctf_zero_crossings(ctf):
    pctf_zeros = []
    pctf = ctf.get_ctf_sine()

    for i in range(0, pctf.shape[0] - 1):
        if pctf[i] > 0 > pctf[i + 1] or pctf[i] < 0 < pctf[i + 1]:
            pctf_zeros.append(i)
            # pctf_zeros.append(ctf.kx[i])

    return pctf_zeros

# ---------------------------------------------------------------

# calc_ctf_1d(1024, 40e-12, ewf_length, defocus=20e-9, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3, fname='pctf1d_new/pctf1d_20nm_inflim2')
# save_range_of_ctf_1d_images(1024, 40e-12, ewf_length, [250e-9, 1040e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# calc_ctf_2d(1024, 40e-12, ewf_length, defocus=0.0, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3, fname='pctf2d_new/pctf2d_0nm_nolab')
# save_range_of_ctf_2d_images(1024, 40e-12, ewf_length, [250e-9, 1040e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# save_range_of_ctf_2d_images(1024, 40e-12, ewf_length, [10e-9, 15e-9, 10e-9], [0.0, 100.5e-9, 1e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# aberrs = ab.Aberrations()
# aberrs.set_C1(20e-9)
# aberrs.set_Cs(0.0)
# aberrs.set_A1(1e-9, np.pi)
# # aberrs.A1.set_am_ph(1e-8, np.pi)
# aberrs.set_df_spread(0.0)
# aberrs.set_conv_angle(0.0)
#
# # ctf = calc_ctf_2d_dev(1024, 40e-12, aberrs, 'ring_test')
# # pctf = ctf.amp * np.sin(ctf.phs)
# # pctf[abs(pctf) > 0.01] = 2
# # pctf[abs(pctf) < 0.01] = 1
# # pctf[pctf==2] = 0
# # save_image(pctf, 'ring_test2.png', -1, 1)
# # 8 nm to 1st ring
#
# ctf = calc_ctf_1d_dev(2048, 40e-12, aberrs)
# kx = ctf.kx * 1e-9
#
# pctf = ctf.get_ctf_sine()
# # pctf[abs(pctf) > 0.01] = 2
# # pctf[abs(pctf) < 0.01] = 1
# # pctf[pctf==2] = 0
#
# for i in range(0, pctf.shape[0]-1):
#     if pctf[i] > 0 > pctf[i+1] or pctf[i] < 0 < pctf[i+1]:
#         print('{0:.2f} nm-1'.format(kx[i]))
#
# # save_range_of_ctf_2d_images_dev(1024, 40e-12, [0.0, 101e-9, 10e-9], aberrs, fname='pctf2d')

# def create_Thon_ring_from_pctf_zeros(ctf, n):
#     # kx, ky = ab.calc_kx_ky(ctf.w, ctf.px)
#     C1 = ctf.abset.get_C1_cf()
#     Cs = ctf.abset.get_Cs_cf()
#     print(Cs)
#     A1 = ctf.abset.get_A1_cf().real
#
#     if Cs > 0:
#         kx = np.sqrt((n * np.pi) / (C1 + A1)) * 1e-9
#         ky = np.sqrt((n * np.pi) / (C1 - A1)) * 1e-9
#         tau = np.arctan2(kx, ky)
#         print(tau * 180.0 / np.pi)
#         print(kx, ky)
#     else:
#         kx2_delta = np.sqrt((C1 + A1) ** 2 + 4 * n * np.pi * Cs)
#         ky2_delta = np.sqrt((C1 - A1) ** 2 + 4 * n * np.pi * Cs)
#
#         kx2_12 = [ (-A1 - C1 + kx2_delta) / (2.0 * Cs), (-A1 - C1 - kx2_delta) / (2.0 * Cs) ]
#         ky2_12 = [ (A1 - C1 + ky2_delta) / (2.0 * Cs), (A1 - C1 - ky2_delta) / (2.0 * Cs) ]
#
#         kx2 = np.max(kx2_12)
#         ky2 = np.max(ky2_12)
#
#         kx = [ np.sqrt(kx2) * 1e-9, -np.sqrt(kx2) * 1e-9 ]
#         ky = [ np.sqrt(ky2) * 1e-9, -np.sqrt(ky2) * 1e-9 ]
#
#         print(kx)
#         print(ky)
#
# aberrs = ab.Aberrations()
# aberrs.set_C1(20e-9)
# aberrs.set_Cs(0.6e-3)
# aberrs.set_A1(1e-9, np.pi)
# # aberrs.A1.set_am_ph(1e-8, np.pi)
# aberrs.set_df_spread(0.0)
# aberrs.set_conv_angle(0.0)
#
# ctf = calc_ctf_2d_dev(1024, 40e-12, aberrs, 'hello')
#
# create_Thon_ring_from_pctf_zeros(ctf, 1)
