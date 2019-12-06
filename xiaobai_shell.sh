source $CONDA_PREFIX/etc/profile.d/conda.sh
tmux new-session -d -s 1 'conda activate cs224w; python train_uci2.py --log_dir c0 --model_types EGSAGE_EGSAGE_EGSAGE --no
de_dim 15 --edge_dim 15 --epochos 200; python load_uci2.py --file c0'