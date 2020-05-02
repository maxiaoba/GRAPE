for seed in 0 1 2 3 4
do
	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.7 --train_y 0.7 --save_model --seed $seed --comment v2miss0.7
done
