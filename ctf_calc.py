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

# calc_ctf_1d(1024, 40e-12, ewf_length, defocus=20e-9, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3, fname='pctf1d_new/pctf1d_20nm_inflim2')
# save_range_of_ctf_1d_images(1024, 40e-12, ewf_length, [250e-9, 1040e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# calc_ctf_2d(1024, 40e-12, ewf_length, defocus=0.0, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3, fname='pctf2d_new/pctf2d_0nm_nolab')
# save_range_of_ctf_2d_images(1024, 40e-12, ewf_length, [250e-9, 1040e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# calc_ctf_2d(1024, 40e-12, ewf_length, defocus=0.0, A1=1e-8+1e-8j, A2=0.0, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3,
#             fname='pctf2d_with_A/pctf2d_0nm')

save_range_of_ctf_2d_images(1024, 40e-12, ewf_length, [10e-9, 15e-9, 10e-9], [0.0, 100.5e-9, 1e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)