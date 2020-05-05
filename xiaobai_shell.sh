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
# 		python baseline_uci_mdi_all.py --best_level --train_edge $train --seed $seed --comment v2train$train
# 	done
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_y_all.py --post_hiddens 0 --train_edge 0.9 --seed $seed --comment v2train0.9
# done

for train in 0.3 0.5 0.7 0.9
do
	for seed in 0 1 2 3 4
	do
		python linear_regression_uci_y_all.py --best_level --train_edge $train --seed $seed --comment v2train$train
	done
done


