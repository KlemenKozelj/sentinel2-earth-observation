import numpy as np
import pandas as pd

def myNDWI_index(eopatch, mask, NDWI_threshold=0.2):

    NDWI = np.copy(eopatch.data['NDWI'].squeeze())
    NDWI[:, ~mask] = float('nan')

    NDWI[~eopatch.mask['VALID_DATA'].squeeze()] = float('nan')
    NDWI = NDWI >= NDWI_threshold

    eo = pd.DataFrame({
        'Date': eopatch.timestamp,
        'NDWI': np.nanmean(NDWI.reshape(len(NDWI), -1), axis=1),
    })

    eo['Date'] = pd.to_datetime(eo['Date']).dt.date
    eo = eo.set_index('Date')

    return eo, NDWI