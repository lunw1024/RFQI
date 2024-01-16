python train_rfqi_susgym.py --data_eps=0.3 --env=BuildingEnv-v0 --max_trn_steps=2000 --eval_freq=50 --device=cuda --data_size=1000000 --batch_size=100 --rho=0.5 --gendata_pol=random

#python eval_rfqi_susgym.py --data_eps=0.3 --gendata_pol=random --env=BuildingEnv-v0 --rho=0.5
