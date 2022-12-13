
##lambda 1, No PreTraining, kits192
```bash
sh fedavg_kits19.sh 6 1 7 nnunet 1000 3 3e-4 kits19 0 0 training 2 0
```
##lambda 4, WIth PreTraining, kits19
```bash
sh fedavg_kits19.sh 6 1 7 nnunet 1000 3 3e-4 kits19 0 0 training 2 1
```
#Lambda 5 Experiments
##Pre-Train variable added, valid sampling, no lr update, kits19
```bash
sh fedavg_kits19.sh 6 1 7 nnunet 1000 3 3e-4 kits19 0 0 training 6 1
```
## CL, RUN ID 4, kits1192
```bash
sh centralized_kits19.sh nnunet 7 3e-4 3000 0 kits19 4 1 training
```

#To be run on lambda 6 
##Pre-Train variable added, valid sampling, no lr update
```bash
sh fedavg_kits19.sh 6 1 7 nnunet 1000 3 3e-4 kits19 0 0 training 5 0
```

##lambda 6 basic, 3000 rounds, ID: 7 (g1lmd2)
```bash
sh fedavg_kits19.sh 6 1 7 nnunet 3000 3 3e-4 kits19 0 0 training 7 0
```