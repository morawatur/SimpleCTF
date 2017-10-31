import numpy as np
import matplotlib.pyplot as plt

import Constants as const
import ImageSupport as imsup

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
    legend = ax.legend(loc='lower right')

    plt.xlim([0, kx[-1]])
    plt.ylim([-1.1, 1.1])
    plt.axhline(0, color='k', lw=1.0)
    plt.xlabel('Spatial frequency k [nm-1]')
    plt.ylabel('Contrast')
    plt.annotate('df = {0:.0f} nm'.format(defocus * 1e9), xy=(0, 0), xytext=(9.5, 0.9), fontsize=16)
    # plt.legend()
    fig.savefig('{0}.png'.format(fname))
    plt.close(fig)

    # pctf_to_save = np.vstack((kx, pctf, spat_env_fun, temp_env_fun)).transpose()
    # np.savetxt('pctf1d.txt', pctf_to_save, delimiter='\t')
    print('Done')
    return pctf

# ---------------------------------------------------------------

def calc_ctf_2d(img_dim, px_dim, ewf_lambda, defocus, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf2d'):
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

    spat_env_fun = np.exp(-((np.pi * conv_angle * kx) ** 2) * (defocus + Cs * ewf_lambda ** 2 * k_squared) ** 2)
    temp_env_fun = np.exp(-(0.5 * np.pi * ewf_lambda * df_spread * k_squared) ** 2)

    pctf *= spat_env_fun
    pctf *= temp_env_fun

    pctf_img = imsup.ImageWithBuffer(img_dim, img_dim)
    pctf_img.LoadAmpData(pctf)
    imsup.SaveAmpImage(pctf_img, '{0}.png'.format(fname))

    print('Done')
    return pctf

# ---------------------------------------------------------------

def save_range_of_ctf_1d_images(img_dim, px_dim, ewf_lambda, df_pars, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf1d'):

    df_min, df_max, df_step = df_pars
    df_values = np.arange(df_min, df_max, df_step)
    for df, idx in zip(df_values, range(df_values.shape[0])):
        fn = '{0}_{1}'.format(fname, idx + 1)
        calc_ctf_1d(img_dim, px_dim, ewf_lambda, df, Cs, df_spread, conv_angle, fn)

    print('All done')

    return

# ---------------------------------------------------------------

def save_range_of_ctf_2d_images(img_dim, px_dim, ewf_lambda, df_pars, Cs=0.0, df_spread=0.0, conv_angle=0.0, fname='pctf2d'):

    df_min, df_max, df_step = df_pars
    df_values = np.arange(df_min, df_max, df_step)
    for df, idx in zip(df_values, range(df_values.shape[0])):
        fn = '{0}_{1}'.format(fname, idx+1)
        calc_ctf_2d(img_dim, px_dim, ewf_lambda, df, Cs, df_spread, conv_angle, fn)

    print('All done')
    return

# ---------------------------------------------------------------

# calc_ctf_1d(1024, 40e-12, const.ewfLambda, defocus=0e-9, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)
save_range_of_ctf_1d_images(1024, 40e-12, const.ewfLambda, [0e-9, 1050e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)

# calc_ctf_2d(1024, 40e-12, const.ewfLambda, defocus=0e-9, Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)
# save_range_of_ctf_2d_images(1024, 40e-12, const.ewfLambda, [0e-9, 1050e-9, 50e-9], Cs=0.6e-3, df_spread=4e-9, conv_angle=0.25e-3)