import numpy as np
from skimage import feature
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def get_water_mask(eopatch, water_treshhold=0.4, canny_sigma=5, gauss_sigma=1):

    water_mask = eopatch.data['NDWI'].squeeze()
    water_mask = water_mask[:, :, :] >= water_treshhold
    water_mask = np.logical_or.reduce(water_mask)

    water_edges = feature.canny(water_mask, sigma=canny_sigma)
    shores = ~gaussian_filter(~water_edges, sigma=gauss_sigma)
    shores_edges = feature.canny(shores, sigma=canny_sigma)
    
    return (water_mask, water_edges, shores, shores_edges)

def visualise_water_mask(band_names, eopatch, index, water_mask, water_edges, shores, shores_edges ):
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))

    fig.suptitle('Timestamp: {}'.format(eopatch.timestamp[index]))

    rgb = np.clip(eopatch.data['BANDS'][index][..., [band_names.index('B04'),band_names.index('B03'),band_names.index('B02')]] * 3, a_min=0, a_max=1)
    ax[0][0].set_title('RGB')
    ax[0][0].imshow(rgb, vmin=0, vmax=1, aspect='auto')

    pos = ax[0][1].imshow(eopatch.data['NDWI'][index].squeeze(), aspect='auto', vmin=-1, vmax=1)
    ax[0][1].set_title('NDWI')
    fig.colorbar(pos, ax=ax[0][1])

    ax[0][2].set_title('Vodna maska')
    ax[0][2].imshow(water_mask, aspect='auto')

    rgb2 = np.copy(rgb)
    rgb2[water_mask] = (1, 0, 0)
    ax[1][0].set_title('RGB z vodno masko')
    ax[1][0].imshow(rgb2, aspect='auto')

    ax[1][1].set_title('Robovi vodne maske')
    ax[1][1].imshow(water_edges, aspect='auto')

    ax[1][2].set_title('Razširjeni robovi vodne maske')
    ax[1][2].imshow(shores, aspect='auto')
