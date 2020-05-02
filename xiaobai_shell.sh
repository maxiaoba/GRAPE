for seed in 0 1 2 3 4
do
	python train_usci_mdi_all.py --post_hiddens 0 --seed $seed --comment v2
done
