// ******* 成员推理攻击，影子数据集 ******* //
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset student --model MLP --train   
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset student --model ResNet --train   
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset obesity --model MLP --train   
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset obesity --model ResNet --train   
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset adult --model MLP --train   
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset adult --model ResNet --train 
python  main_lira.py --gpu 0 --evaluate_type 0 --dataset adult --model TabNet --train  

// ******* HDBSCAN聚类分析 ******* //
// Student 4424 //
python  main_lira.py --gpu 0 --evaluate_type 1 --dataset student --model MLP --train   

python  main_lira.py --gpu 0 --evaluate_type 1 --dataset student --model ResNet --train   

// Obesity 2111 //
python  main_lira.py --gpu 0 --evaluate_type 1 --dataset obesity --model MLP --train   

python  main_lira.py --gpu 0 --evaluate_type 1 --dataset obesity --model ResNet --train   

// Adult 48842 //
python  main_lira.py --gpu 0 --evaluate_type 1 --dataset adult --model MLP --train   

python  main_lira.py --gpu 0 --evaluate_type 1 --dataset adult --model ResNet --train   

// ******* QID脆弱性分析 ******* //
python  main_lira.py --gpu 0 --evaluate_type 2 --dataset student
python  main_lira.py --gpu 0 --evaluate_type 2 --dataset obesity
python  main_lira.py --gpu 0 --evaluate_type 2 --dataset adult

// ******* OPRS ******* //
python  main_lira.py --gpu 0 --evaluate_type 3 --dataset student --model MLP --train   
python  main_lira.py --gpu 0 --evaluate_type 3 --dataset obesity --model MLP --train   
python  main_lira.py --gpu 0 --evaluate_type 3 --dataset adult --model MLP --train   

python  main_lira.py --gpu 0 --evaluate_type 3 --dataset student --model ResNet --train   
python  main_lira.py --gpu 0 --evaluate_type 3 --dataset obesity --model ResNet --train   
python  main_lira.py --gpu 0 --evaluate_type 3 --dataset adult --model ResNet --train   