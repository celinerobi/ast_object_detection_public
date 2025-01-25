This repository containsus used to train an object detection model to detect Aboveground Storage Tanks (AST) using the data created by [Robinson et al, 2024](https://www.nature.com/articles/s41597-023-02780-1). 
The model is applied to detect ASTs in areas prone to storm surge. 

1. The `data.sh` script formats the dataset is formatted into the yolov8 format.
2. The `tune.sh` script uses genetic algorithms to optimize model parameters.
3. The `train.sh` script trains the yolov8 model.
4. The `val.sh` script validates the model. 
5. The `obtain_aerial_imagery.sh` script is used to download and process aerial imagery in the case study area for prediction.  
- **Extracting Data in the Study Area (`naip_in_slosh.sh`)**: Extracts imagery data relevant to the defined study area to narrow the focus for prediction.  
- **Downloading Data for Prediction (`download.sh`)**: Downloads aerial imagery data from Microsoft Planetary Computer.
- **Chipping Tiles into Smaller Images for Prediction (`chip_tiles.sh`)**: Divides large imagery tiles into smaller, manageable images that meet the input requirements for the prediction model.  
6. Prediction Pipeline
Run predictions for AST detection in the study area using the complete_prediction.sh script. This script orchestrates the full prediction workflow, including:
- **Running Predictions (`predict.sh`)**: Performing object detection on batched images.
- **Estimating Tank Heights (`height_estimation.sh`)**: Calculates the height of detected objects using LiDAR.
- **Compiling Predictions (`compile_predictions.sh`)**: Cleaning and merging prediction results into a complete dataset.