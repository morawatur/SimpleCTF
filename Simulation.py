import numpy as np

import Dm3Reader3 as dm3
import ImageSupport as imsup
import Propagation as prop

# ---------------------------------------------------------------

def simulate_image_for_defocus(img, defocus):
    ctf = prop.CalcTransferFunction(img.width, img.px_dim, defocus)
    ctf.AmPh2ReIm()
    img_sim = prop.PropagateWave(img, ctf)
    img_sim.ReIm2AmPh()
    img_sim.MoveToCPU()
    return img_sim

# ---------------------------------------------------------------

img_data, px_dims = dm3.ReadDm3File('df_sim0.dm3')
px_dims[0] = 40e-12
imsup.Image.px_dim_default = px_dims[0]
img_data = np.abs(img_data)
img = imsup.ImageWithBuffer(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'], 1, px_dim_sz=px_dims[0])
print(img.px_dim)
img.LoadAmpData(np.sqrt(img_data).astype(np.float32))

img_sim = simulate_image_for_defocus(img, 10e-9)
imsup.DisplayAmpImage(img_sim)
imsup.DisplayPhaseImage(img_sim)


