#(trap 'kill 0' SIGINT;
#python train_uci_y_all.py --comment v2 --seed 0 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 1 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 2 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 3 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 4 &
#sleep 10
#python train_uci_mdi_all.py --comment v2 --seed 0 &
#sleep 10
#python train_uci_mdi_all.py --comment v2 --seed 1 &
#sleep 10
#python train_uci_mdi_all.py --comment v2 --seed 2 &
#sleep 10
#python train_uci_mdi_all.py --comment v2 --seed 3 &
#sleep 10
#python train_uci_mdi_all.py --comment v2 --seed 4
#)


(trap 'kill 0' SIGINT;
python train_uci_y_all.py --opt_scheduler cos --comment v2_cos --seed 0 &
sleep 10
python train_uci_y_all.py --opt_scheduler cos --comment v2_cos --seed 1 &
sleep 10
python train_uci_y_all.py --opt_scheduler cos --comment v2_cos --seed 2 &
sleep 10
python train_uci_y_all.py --opt_scheduler cos --comment v2_cos --seed 3 &
sleep 10
python train_uci_y_all.py --opt_scheduler cos --comment v2_cos --seed 4 &
sleep 10
python train_uci_mdi_all.py --opt_scheduler cos --comment v2_cos --seed 0 &
sleep 10
python train_uci_mdi_all.py --opt_scheduler cos --comment v2_cos --seed 1 &
sleep 10
python train_uci_mdi_all.py --opt_scheduler cos --comment v2_cos --seed 2 &
sleep 10
python train_uci_mdi_all.py --opt_scheduler cos --comment v2_cos --seed 3 &
sleep 10
python train_uci_mdi_all.py --opt_scheduler cos --comment v2_cos --seed 4
)







#
#(trap 'kill 0' SIGINT;
#python train_uci_mdi_all.py --comment v2 --seed 0 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 1 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 2 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 3 &
#sleep 10
#python train_uci_y_all.py --comment v2 --seed 4 &
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

#
#(trap 'kill 0' SIGINT;
#python linear_regression_uci_y.py --comment v1 --seed 0 &
#sleep 10
#python linear_regression_uci_y.py --comment v1 --seed 1 &
#sleep 10
#python linear_regression_uci_y.py --comment v1 --seed 2 &
#sleep 10
#python linear_regression_uci_y.py --comment v1 --seed 3 &
#sleep 10
#python linear_regression_uci_y.py --comment v1 --seed 4
#)