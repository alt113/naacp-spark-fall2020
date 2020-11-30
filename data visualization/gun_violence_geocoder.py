import pandas as pd
# import geopandas as gpd
import geopy
from geopy.extra.rate_limiter import RateLimiter


gv_df = pd.read_csv("gun_violence_incidents_2014-2018.csv")
# gv_df = gv_df[:100]

geolocator = geopy.Nominatim(user_agent="myGeocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds = 1)

gv_df['Full Address'] = gv_df['Address'] + "," + gv_df['City Or County'] + "," + gv_df['State']

gv_df['Location'] = gv_df['Full Address'].apply(geocode)

gv_df['Point'] = gv_df['Location'].apply(lambda loc: tuple(loc.point) if loc else None)

not_none = gv_df['Point'] == gv_df['Point']
gv_df_filtered = gv_df[not_none]

gv_df_filtered[['Latitude', 'Longitude', 'Altitude']] = pd.DataFrame(gv_df_filtered['Point'].tolist(), index=gv_df_filtered.index)
print(gv_df_filtered)
gv_df_filtered.to_csv("gv_incidents_geocoded.csv", index=False)
