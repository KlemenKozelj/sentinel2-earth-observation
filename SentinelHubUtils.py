from sentinelhub import SHConfig
from os import environ
from datetime import timedelta
import numpy as np

sh_config = SHConfig()
sh_config.sh_client_id=environ['SH_CLIENT_ID']
sh_config.sh_client_secret=environ['SH_CLIENT_SECRET']

from eolearn.core import EOTask, EOPatch, LinearWorkflow, Dependency, FeatureType, OverwritePermission
from eolearn.io.processing_api import SentinelHubInputTask
from eolearn.core import LoadFromDisk, SaveToDisk
from eolearn.mask import AddValidDataMaskTask
from eolearn.features import SimpleFilterTask, NormalizedDifferenceIndexTask
from eolearn.geometry import VectorToRaster

from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from skimage.filters import threshold_otsu
import shapely.wkt
from shapely.geometry import Polygon
from sentinelhub import BBox, CRS, DataCollection

data_level = DataCollection.SENTINEL2_L2A
band_names = ['B02','B03','B04','B08','B8A','B11','B12']

if data_level.api_id == 'S2L1C':
    sh_config.instance_id='792138b2-7347-4b22-b90b-ae2089b83cf2'
else:
    sh_config.instance_id='9712c39b-d56b-4133-bee7-573295d0e478'
    if 'B10' in band_names: band_names.remove('B10')

cloud_tolerance = 0.1

download_task = SentinelHubInputTask(
    data_collection=data_level,
    bands_feature=(FeatureType.DATA, 'BANDS'),
    resolution=10,
    maxcc=cloud_tolerance,
    cache_folder='./.cache',
    bands=band_names,
    time_difference=timedelta(hours=24),
    additional_data=[
        (FeatureType.MASK, 'dataMask', 'IS_DATA'),
        (FeatureType.MASK, 'CLM'),
        (FeatureType.MASK, 'CLP'),
    ],
    config=sh_config
)

calculate_ndvi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDVI'), (band_names.index('B8A'), band_names.index('B04')))
calculate_ndwi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDWI'), (band_names.index('B03'), band_names.index('B8A')))

def calculate_valid_data_mask(eopatch):
    is_data_mask = eopatch.mask['IS_DATA'].astype(np.bool)
    cloud_mask = ~eopatch.mask['CLM'].astype(np.bool)
    return np.logical_and(is_data_mask, cloud_mask)

add_valid_mask = AddValidDataMaskTask(predicate=calculate_valid_data_mask)

def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)

class AddValidDataCoverage(EOTask):

    def execute(self, eopatch):
        valid_data = eopatch.get_feature(FeatureType.MASK, 'VALID_DATA')
        time, height, width, channels = valid_data.shape
        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height * width * channels)))
        eopatch.add_feature(FeatureType.SCALAR, 'COVERAGE', coverage[:, np.newaxis])
        return eopatch

add_coverage = AddValidDataCoverage()

class ValidDataCoveragePredicate:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold

remove_cloudy_scenes = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'), ValidDataCoveragePredicate(cloud_tolerance))

def get_eopatch(dirname, geo_points, time_interval):
    eopatch = EOPatch.load(dirname)
    if len(eopatch.data) == 0:
        print('Downloading eopatch data from Sentinel hub')

        workflow = LinearWorkflow(
            download_task,
            calculate_ndwi,
            calculate_ndvi,
            add_valid_mask,
            add_coverage,
            remove_cloudy_scenes,
        )

        roi_bbox = BBox(bbox=[
            geo_points[0][1],
            geo_points[0][0],
            geo_points[1][1],
            geo_points[1][0]],
            crs=CRS.WGS84
        )
        
        result = workflow.execute({ download_task: { 'bbox': roi_bbox, 'time_interval': time_interval } })
        eopatch = result.eopatch()
        eopatch.save(dirname, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
    else:
        print('eopatch data was loaded from local directory')
    return eopatch

def delete_frame_eopatch(eopatch, index):
    eopatch.data['BANDS'] = np.delete(eopatch.data['BANDS'], index, axis=0)
    eopatch.data['NDVI'] = np.delete(eopatch.data['NDVI'], index, axis=0)
    eopatch.data['NDWI'] = np.delete(eopatch.data['NDWI'], index, axis=0)
    eopatch.mask['CLM'] = np.delete(eopatch.mask['CLM'], index, axis=0)
    eopatch.mask['CLP'] = np.delete(eopatch.mask['CLP'], index, axis=0)
    eopatch.mask['IS_DATA'] = np.delete(eopatch.mask['IS_DATA'], index, axis=0)
    eopatch.mask['VALID_DATA'] = np.delete(eopatch.mask['VALID_DATA'], index, axis=0)
    eopatch.scalar['COVERAGE'] = np.delete(eopatch.scalar['COVERAGE'], index, axis=0)
    eopatch.timestamp.pop(index)
    return eopatch
