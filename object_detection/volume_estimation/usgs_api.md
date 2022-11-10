# api access
https://ers.cr.usgs.gov/profile/access
https://apps.nationalmap.gov/tnmaccess/
http get in python
https://realpython.com/python-requests/
https://www.w3schools.com/tags/ref_httpmethods.asp
https://stackoverflow.com/questions/645312/what-is-the-quickest-way-to-http-get-in-python

# Lidar Data
select the file with the largest bounding box/that is newest
matches file type "LAS,LAZ"

# DEM Data
select the file with the largest bounding box/that is newest
matches file type "LAS,LAZ"

- Pull Lidar and Elevation data from USGS
#sbatch usgs_tnm_api.sh #'Lidar Point Cloud (LPC)'

- Subset AST dataset where both LPC and DEM data is available
- Download highest resolution LPC and DEM Data available 
- Estimate Heights

sbatch lidar_subset_by_tank.sh
DEM_by_tank.py

- height estimation and  Plots
image_by_tank.py #
height_estimation_and_plot.sh
