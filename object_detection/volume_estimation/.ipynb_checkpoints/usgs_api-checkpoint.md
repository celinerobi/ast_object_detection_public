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

"Digital Elevation Model (DEM) 1 meter"
"National Elevation Dataset (NED) 1/9 arc-second"

python cred/AST_dataset/object_detection/volume_estimation/usgs_
tnm_api.py --tile_level_annotations_path "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//compl
ete_dataset_/tile_level_annotations/tile_level_annotations.geojson" --dataset_name "Lidar Point Clo
ud (LPC)" --request_total_idx "total" --request_content_idx "items" --request_content_names_idx "ti
tle"

--stored_data_path