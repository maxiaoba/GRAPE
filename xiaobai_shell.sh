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

# for train in 0.3 0.5 0.7 0.9
# do
# 	for seed in 0 1 2 3 4
# 	do
# 		python linear_regression_uci_y_all.py --best_level --train_edge $train --seed $seed --comment v2train$train
# 	done
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --known 1. --seed $seed --comment v2train0.7known1.0
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --split_sample 0.7 --split_test --seed $seed --comment v2train0.7split0.7test
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --split_sample 0.7 --split_train --split_test --seed $seed --comment v2train0.7split0.7traintest
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --best_level --split_sample 0.7 --seed $seed --comment v2train0.7split0.7
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --best_level --split_sample 0.7 --split_test --seed $seed --comment v2train0.7split0.7test
# done

# for seed in 0 1 2 3 4
# do
# 	python baseline_uci_mdi_all.py --best_level --split_sample 0.7 --split_by random --seed $seed --comment v2train0.7splitrandom0.7
# 	python baseline_uci_mdi_all.py --best_level --split_sample 0.7 --split_by random --split_test --seed $seed --comment v2train0.7splitrandom0.7test
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --split_sample 0.7 --split_by random --split_train --split_test --seed $seed --comment v2train0.7splitrandom0.7traintest
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --split_sample 0.7 --split_by random --split_train --seed $seed --comment v2train0.7splitrandom0.7train
# done

# for seed in 0 1 2 3 4
# do
# 	python train_uci_mdi_all.py --post_hiddens 0 --save_model --split_sample 0.7 --split_by random --split_test --seed $seed --comment v2train0.7splitrandom0.7test
# done

for seed in 0 1 2 3 4
do
	python linear_regression_uci_y_all.py --best_level --train_edge 0.7 --seed $seed --comment v2train0.7
done

