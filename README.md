# Crowd Counting

### generate density map with ground truth
 - `data preprocess (shtu, ucf_qnrf, ucf_cc_50)`
    - edit the `src/param.json` to set the path of datasets
    - run `src/data_preprocess/datasets.py` to auto generate density map

### get datasets
 - edit `src/datasets`
 
### get models
 - edit `src/models` like `cbam_net`

### utils
 - `src/utils` for loss function
 
### how to start
 - run `sh main.sh model_name dataset_name cuda_index` like `sh main.sh cbam_net shtu_dataset 3`