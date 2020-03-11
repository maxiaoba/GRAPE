#(trap 'kill 0' SIGINT;
#python train_uci_y.py --log_dir gnn_v1 --repeat 0 &
#sleep 10
#python train_uci_y.py --log_dir gnn_v1 --repeat 1 &
#sleep 10
#python train_uci_y.py --log_dir gnn_v1 --repeat 2 &
#sleep 10
#python train_uci_y.py --log_dir gnn_v1 --repeat 3 &
#sleep 10
#python train_uci_y.py --log_dir gnn_v1 --repeat 4
#)

#(trap 'kill 0' SIGINT;
#python train_uci_mdi.py --comment v1 --seed 0 &
#sleep 10
#python train_uci_mdi.py --comment v1 --seed 1 &
#sleep 10
#python train_uci_mdi.py --comment v1 --seed 2 &
#sleep 10
#python train_uci_mdi.py --comment v1 --seed 3 &
#sleep 10
#python train_uci_mdi.py --comment v1 --seed 4
#)


(trap 'kill 0' SIGINT;
python linear_regression_uci_y.py --comment v1 --seed 0 &
sleep 10
python linear_regression_uci_y.py --comment v1 --seed 1 &
sleep 10
python linear_regression_uci_y.py --comment v1 --seed 2 &
sleep 10
python linear_regression_uci_y.py --comment v1 --seed 3 &
sleep 10
python linear_regression_uci_y.py --comment v1 --seed 4
)