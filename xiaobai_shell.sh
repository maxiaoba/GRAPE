# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.7 --train_y 0.7 --save_model --seed $seed --comment v2miss0.7
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.5 --train_y 0.5 --save_model --seed $seed --comment v2miss0.5
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.9 --train_y 0.9 --save_model --seed $seed --comment v2miss0.9
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.3 --train_y 0.3 --save_model --seed $seed --comment v2miss0.3
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 0 --seed $seed --comment v2lv0
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 1 --seed $seed --comment v2lv1
# done

for seed in 0 1 2 3 4
do
	python baseline_uci_mdi_all.py --level 2 --seed $seed --comment v2lv2
done
