import warnings
import shapely
import shapely.geometry
from shapely.geometry import Polygon
from shapely.ops import orient

from rasterio.warp import transform_geom
from ..coords import _utm_zone, CRS_LATLON


class Feature:
    """
    Proxy class for shapely geometry, include crs and properties of feature
    """

    def __init__(self, geometry, properties=None, crs=CRS_LATLON):
        self.crs = crs
        self._geometry = self._valid(
            shapely.geometry.shape(geometry))
        self.properties = properties

    def __repr__(self):
        print('CRS: {}\nProperties: {}'.format(self.crs, self.properties))
        return repr(self._geometry)

    def __getattr__(self, item):
        return getattr(self._geometry, item)
    
    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def _valid(self, shape):
        if not shape.is_valid:
            shape = shape.buffer(0)
        return shape

    def apply(self, func):
        return Feature(func(self._geometry), properties=self.properties, crs=self.crs)

    @property
    def shape(self):
        return self._geometry

    @property
    def geometry(self):
        return shapely.geometry.mapping(self._geometry)

    def as_geojson(self, hold_crs=False):
        """ Return Feature as GeoJSON formatted dict
        Args:
            hold_crs (bool): serialize with current projection, that could be not ESPG:4326 (which is standards violation)
        Returns:
            GeoJSON formatted dict
        """
        if self.crs != CRS_LATLON and not hold_crs:
            f = self.reproject(CRS_LATLON)
        else:
            f = self

        shape = f.shape
        if shape.is_empty:
            # Empty geometries are not allowed in FeatureCollections,
            # but here it may occur due to reprojection which can eliminate small geiometries
            # This case is processed separately as orient(POLYGON_EMPTY) raises an exception
            # TODO: do not return anything on empty polygon and ignore such features in FeatureCollection.geojson
            shape = Polygon()
        else:
            try:
                shape = orient(shape)
            except Exception as e:
                # Orientation is really not a crucial step, it follows the geojson standard,
                # but not oriented polygons can be read by any instrument. So, ni case of any troubles with orientation
                # we just fall back to not-oriented version of the same geometry
                warnings.warn(f'Polygon orientation failed: {str(e)}. Returning initial shape instead',
                              RuntimeWarning)
                shape = f.shape

        f = Feature(shape, properties=f.properties)
        data = {
            'type': 'Feature',
            'geometry': f.geometry,
            'properties': f.properties
        }
        return data

    @property
    def geojson(self):
        return self.as_geojson()

    def reproject(self, dst_crs):
        new_geometry = transform_geom(
            src_crs=self.crs,
            dst_crs=dst_crs,
            geom=self.geometry,
        )
        return Feature(new_geometry, properties=self.properties, crs=dst_crs)
    
    def reproject_to_utm(self):
        lon1, lat1, lon2, lat2 = self.shape.bounds
        utm_zone = _utm_zone((lat1 + lat2)/2 , (lon1 + lon2)/2)
        return self.reproject(utm_zone)

