# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 0 --seed $seed --comment v2lv0
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 1 --seed $seed --comment v2lv1
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 1 --seed $seed --comment v2lv1
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --level 0 --seed $seed --comment v2lv0
# done

# for train in 0.3 0.5 0.7 0.9
# do
# 	for seed in 0 1 2 3 4
# 	do
# 		python baseline_uci_mdi_all.py --best_level --train_edge $train --seed $seed --comment v2_best_train$train
# 	done
# done

for seed in 0 1 2 3 4
do
	python train_uci_mdi_all.py --post_hiddens 0 --train_edge 0.5 --epochs 50000 --valid 0.1 --save_model --seed $seed --comment ep5e4_v2_train0.5
done


