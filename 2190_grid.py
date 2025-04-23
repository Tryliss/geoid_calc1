import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from geoid_toolkit.read_ICGEM_harmonics import read_ICGEM_harmonics
from geoid_toolkit.topographic_potential import topographic_potential
from geoid_toolkit.real_potential import real_potential
from geoid_toolkit.norm_potential import norm_potential
from geoid_toolkit.norm_gravity import norm_gravity
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pykrige.ok import OrdinaryKriging
import matplotlib.ticker as ticker

# Parameters
lmax = 2190
R = 6378136.3
GM = 3.986004415E+14

# Define the latitude and longitude ranges with 5 km spacing
# Approximation: 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 111 km * cos(latitude)
lat_spacing = 5 / 111  # 5 km in degrees latitude
lon_spacing = 5 / (111 * np.cos(np.radians(39.5)))  # 5 km in degrees longitude (approx. at the center latitude)

lat_range = np.arange(35, 44 + lat_spacing, lat_spacing)  # Latitude range with 5 km spacing
lon_range = np.arange(-10, 4.5 + lon_spacing, lon_spacing)  # Longitude range with 5 km spacing

# Create a meshgrid of latitudes and longitudes
lats, lons = np.meshgrid(lat_range, lon_range)

# Flatten the meshgrid for easier iteration
flat_lats = lats.flatten()
flat_lons = lons.flatten()

# Extracting static gravity model file coefficients
gravity_model_file = 'SGG-UGM-2.gfc'
Ylms = read_ICGEM_harmonics(gravity_model_file, TIDE='zero_tide', lmax=lmax, ELLIPSOID='WGS84')
clm = Ylms['clm']
slm = Ylms['slm']

# Read topography harmonics
def read_topography_harmonics(model_file):
    dinput = np.fromfile(model_file, dtype=np.dtype('<f8'))
    header = 2
    input_lmin, input_lmax = dinput[:header].astype(np.int64)
    n_down = ((input_lmin - 1)**2 + 3*(input_lmin - 1)) // 2 + 1
    n_up = (input_lmax**2 + 3*input_lmax) // 2 + 1
    n_harm = n_up - n_down
    model_input = {}
    model_input['modelname'] = 'EARTH2014'
    model_input['density'] = 2670.0
    ii, jj = np.tril_indices(input_lmax + 1)
    model_input['l'] = np.arange(input_lmax + 1)
    model_input['m'] = np.arange(input_lmax + 1)
    model_input['clm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['slm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['clm'][ii, jj] = dinput[header:(header + n_harm)]
    model_input['slm'][ii, jj] = dinput[(header + n_harm):(header + 2 * n_harm)]
    return model_input

model_file = 'dV_ELL_EARTH2014_5480.bshc'
model_input = read_topography_harmonics(model_file)
tclm = model_input['clm']
tslm = model_input['slm']
density = model_input['density']

# Function to compute corrected geoid undulation
def corrected_geoid_undulation(lat, lon, refell, clm, slm, tclm, tslm, lmax, R, GM, density, GAUSS=0, EPS=1e-8, max_iterations=1000):
    try:
        # Ensure lat and lon are arrays
        lat = np.array([lat])  # Wrap scalar lat in an array
        lon = np.array([lon])  # Wrap scalar lon in an array

        # Compute real potential
        W, dWdr = real_potential(lat, lon, 0.0, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)

        # Compute normal potential
        U, dUdr, dUdt = norm_potential(lat, lon, 0.0, refell, lmax)

        # Compute topographic potential correction
        T = topographic_potential(lat, lon, refell, tclm, tslm, lmax, R, density)

        # Compute normal gravity at latitude
        gamma_h, dgamma_dh = norm_gravity(lat, 0.0, refell)

        # Initial geoid height
        N_1 = (W - U - T) / gamma_h
        N = np.copy(N_1)
        RMS = np.inf
        iteration = 0

        # Iterative refinement
        while RMS > EPS and iteration < max_iterations:
            W, dWdr = real_potential(lat, lon, N_1, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
            N_1 += (W - U - T) / gamma_h
            RMS = np.sqrt(np.sum((N - N_1)**2) / len(lat))
            N = np.copy(N_1)
            iteration += 1

        return N.item()  # Extract scalar value from the returned array

    except Exception as e:
        print(f"Error in worker process for lat={lat}, lon={lon}: {e}")
        return np.nan  # Return NaN for failed computations

# Parallel computation of geoid undulation heights
def compute_geoid_parallel(flat_lats, flat_lons, refell, clm, slm, tclm, tslm, lmax, R, GM, density):
    # Use partial to pre-fill the additional arguments
    partial_function = partial(corrected_geoid_undulation, refell=refell, clm=clm, slm=slm, tclm=tclm, tslm=tslm, lmax=lmax, R=R, GM=GM, density=density)
    with ProcessPoolExecutor(max_workers=20) as executor:  # Limit to 4 workers to reduce memory usage
        results = list(tqdm(executor.map(partial_function, flat_lats, flat_lons), total=len(flat_lats), desc="Processing points in parallel"))
    return np.array(results)

# Compute geoid undulation heights in parallel
print("Computing geoid undulation heights in parallel...")
geoid_heights = compute_geoid_parallel(flat_lats, flat_lons, 'WGS84', clm, slm, tclm, tslm, lmax, R, GM, density)

# Save the results to a CSV file
output_file = 'geoid_undulation_spain.csv'
results_df = pd.DataFrame({
    'Latitude': flat_lats,
    'Longitude': flat_lons,
    'Geoid_Height': geoid_heights
})
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Load the geoid undulation data from the CSV file
csv_file = 'geoid_undulation_spain.csv'
results_df = pd.read_csv(csv_file)

# Extract the latitude, longitude, and geoid height data
flat_lats_original = results_df['Latitude'].values
flat_lons_original = results_df['Longitude'].values
geoid_heights = results_df['Geoid_Height'].values

# Perform kriging interpolation with Gaussian variogram model
kriging_model = OrdinaryKriging(
    flat_lons_original, flat_lats_original, geoid_heights,
    variogram_model="gaussian",  # Use Gaussian variogram model
    verbose=False,
    enable_plotting=False
)

# Define the finer grid for interpolation with 500 points
fine_lon_range = np.linspace(-10, 4.5, 500)  # 500 interpolated longitude points
fine_lat_range = np.linspace(35, 44, 500)  # 500 interpolated latitude points
fine_lons, fine_lats = np.meshgrid(fine_lon_range, fine_lat_range)

# Interpolate geoid heights on the finer grid
fine_geoid_heights, _ = kriging_model.execute("grid", fine_lon_range, fine_lat_range)

# Plot the interpolated geoid undulation heights
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([-10, 4.5, 35, 44], crs=ccrs.PlateCarree())  # Set the extent to include -10 longitude

# Add features to the map
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')  # Country borders
ax.add_feature(cfeature.COASTLINE, edgecolor='black')  # Coastlines
ax.add_feature(cfeature.LAND, facecolor='lightgray')  # Land
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # Ocean

# Add latitude and longitude labels on the axes
ax.set_xticks(np.arange(-10, 5, 2), crs=ccrs.PlateCarree())  # Longitude ticks
ax.set_yticks(np.arange(35, 45, 2), crs=ccrs.PlateCarree())  # Latitude ticks
ax.set_xticklabels([f"{lon}°" for lon in np.arange(-10, 5, 2)], fontsize=10)  # Longitude labels
ax.set_yticklabels([f"{lat}°" for lat in np.arange(35, 45, 2)], fontsize=10)  # Latitude labels

# Plot the geoid undulation heights
contour = ax.contourf(
    fine_lon_range, fine_lat_range, fine_geoid_heights,
    transform=ccrs.PlateCarree(),
    cmap='RdYlGn_r',  # Green to red color map
    levels=np.arange(np.floor(fine_geoid_heights.min()), np.ceil(fine_geoid_heights.max()) + 0.09, 0.09)  # Integer levels
)

# Add color bar with integer ticks
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='Undulation (m)')
cbar.locator = ticker.MaxNLocator(integer=True)  # Ensure integer ticks on the color bar
cbar.update_ticks()  # Update the color bar to apply the integer ticks

# Add title
plt.title('Geoid undulation heights for Spanish mainland and Balearic Islands')

# Add copyright notice inside the map
ax.text(
    4.3, 35.2,  # Position: slightly inside the bottom-right corner of the map
    '© cpalomar',  # Copyright text
    fontsize=10, color='black', ha='right', va='bottom', transform=ccrs.PlateCarree()
)

# Save the plot
plot_file = 'geoid_undulation_map.png'
plt.savefig(plot_file)
print(f"Map saved to {plot_file}")

# Uncomment to display the plot
# plt.show()
