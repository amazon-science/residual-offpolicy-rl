# Generate the image dataset for the transport/mh dataset
python robomimic/scripts/dataset_states_to_obs.py --dataset /mnt/mlfast/robomimic/transport/mh/low_dim_v15.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names shouldercamera0_image shouldercamera1_image --camera_height 84 --camera_width 84