import numpy as np
import matplotlib.pyplot as plt

import Constants as const
import ImageSupport as imsup

# ---------------------------------------------------------------

def calc_ctf_1d(img_dim, px_dim, ewf_lambda, defocus, Cs=0.0, df_spread=0.0, conv_angle=0.0):
    df_coeff = np.pi * ewf_lambda * defocus
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs

    rec_px_dim = 1.0 / (img_dim * px_dim)

    x = np.arange(0, img_dim // 2, 1)
    kx = x * rec_px_dim
    k_squared = kx ** 2

    aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)
    pctf = np.sin(aberr_fun)

    spat_env_fun = np.exp(-(np.pi * conv_angle * defocus * kx) ** 2)
    temp_env_fun = np.exp(-(np.pi * ewf_lambda * df_spread * (kx ** 2)) ** 2)

    pctf *= spat_env_fun
    pctf *= temp_env_fun

    kx *= 1e-9
    fig, ax = plt.subplots()
    ax.plot(kx, pctf, 'r', label='PCTF')
    ax.plot(kx, spat_env_fun, 'g', label='Spatial envelope')
    ax.plot(kx, temp_env_fun, 'b', label='Temporal envelope')
    # legend = ax.legend(loc='lower right', shadow=True)

    plt.xlim([0, kx[-1]])
    plt.ylim(bottom=-1)
    plt.axhline(0, color='k', lw=1.0)
    plt.xlabel('k [nm-1]')
    plt.legend()
    # plt.show()
    fig.savefig('foo.png', dpi=200)

    # pctf_to_save = np.vstack((kx * 1e-9, pctf, spat_env_fun, temp_env_fun)).transpose()
    # np.savetxt('pctf1d.txt', pctf_to_save, delimiter='\t')
    print('Done')
    return pctf

# ---------------------------------------------------------------

def calc_ctf_2d(img_dim, px_dim, ewf_lambda, defocus, Cs=0.0):
    df_coeff = np.pi * ewf_lambda * defocus
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs

    rec_px_dim = 1.0 / (img_dim * px_dim)
    rec_orig = -1.0 / (2.0 * px_dim)

    x, y = np.mgrid[0:img_dim:1, 0:img_dim:1]
    kx = rec_orig + x * rec_px_dim
    ky = rec_orig + y * rec_px_dim
    k_squared = kx ** 2 + ky ** 2

    aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)

    pctf = np.sin(aberr_fun)

    pctf_img = imsup.ImageWithBuffer(img_dim, img_dim)
    pctf_img.LoadAmpData(pctf)
    imsup.SaveAmpImage(pctf_img, 'pctf2d.png')

    print('Done')
    return pctf

# ---------------------------------------------------------------

def save_range_of_ctf_images(img_dim, px_dim, ewf_lambda, df_pars, Cs=0.0, df_spread=0.0, conv_angle=0.0):
    df_min, df_max, df_step = df_pars
    Cs_coeff = 0.5 * np.pi * (ewf_lambda ** 3) * Cs
    rec_px_dim = 1.0 / (img_dim * px_dim)
    x = np.arange(0, img_dim // 2, 1)
    kx = x * rec_px_dim
    k_squared = kx ** 2
    temp_env_fun = np.exp(-0.5 * ((np.pi * ewf_lambda * df_spread) ** 2) * (kx ** 4))       # 0.5 czy nie?
    f_idx = 0

    for df in np.arange(df_min, df_max, df_step):
        f_idx += 1
        df_coeff = np.pi * ewf_lambda * df
        aberr_fun = df_coeff * k_squared + Cs_coeff * (k_squared ** 2)
        pctf = np.sin(aberr_fun)
        spat_env_fun = np.exp(-(np.pi * conv_angle * df * kx) ** 2)
        pctf *= spat_env_fun
        pctf *= temp_env_fun

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(kx * 1e-9, pctf, 'r', label='PCTF')
        ax.plot(kx * 1e-9, spat_env_fun, 'g', label='Spatial envelope')
        ax.plot(kx * 1e-9, temp_env_fun, 'b', label='Temporal envelope')
        legend = ax.legend(loc='lower right')

        plt.xlim([0, kx[-1] * 1e-9])
        plt.ylim([-1, 1.1])
        plt.axhline(0, color='k', lw=1.0)
        plt.xlabel('Spatial frequency k [nm-1]')
        plt.ylabel('Contrast')
        plt.annotate('df = {0:.0f} nm'.format(df * 1e9), xy=(0, 0), xytext=(11, 1))
        # plt.legend()
        fig.savefig('foo{0}.png'.format(f_idx))

    print('Done')
    return

# ---------------------------------------------------------------

save_range_of_ctf_images(1024, 40e-12, const.ewfLambda, [0e-9, 100e-9, 10e-9], Cs=0.6e-3, df_spread=3e-9, conv_angle=1e-3)