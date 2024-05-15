//<MemInf,MB,DS>,黑盒+影子
python main.py --gpu 0 --evaluate_type 0 --dataset student --mode 0 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset student --mode 0 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset obesity --mode 0 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset obesity --mode 0 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset adult --mode 0 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset adult --mode 0 --model ResNet --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 0 --dataset student --mode 1 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset student --mode 1 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset obesity --mode 1 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset obesity --mode 1 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset adult --mode 1 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 0 --dataset adult --mode 1 --model ResNet --train_target --train_shadow

// ******* HDBSCAN ******* //
// *** Student 4424 ***//
python main.py --gpu 0 --evaluate_type 1 --dataset student --mode 0 --model MLP --kmeans
python main.py --gpu 0 --evaluate_type 1 --dataset student --mode 0 --model MLP --kmeans --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 1 --dataset student --mode 1 --model MLP --kmeans --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 1 --dataset student --mode 0 --model ResNet --kmeans --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 1 --dataset student --mode 1 --model ResNet --kmeans --train_target --train_shadow

// *** Obesity 2111 ***//
python main.py --gpu 0 --evaluate_type 1 --dataset obesity --mode 0 --model MLP --kmeans --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 1 --dataset obesity --mode 1 --model MLP --kmeans --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 1 --dataset obesity --mode 0 --model ResNet --kmeans --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 1 --dataset obesity --mode 1 --model ResNet --kmeans --train_target --train_shadow

// *** Adult 48842 ***//
python main.py --gpu 0 --evaluate_type 1 --dataset adult --mode 0 --model MLP --kmeans --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 1 --dataset adult --mode 0 --model ResNet --kmeans --train_target --train_shadow

// ******* QID ******* //
python main.py --gpu 0 --evaluate_type 2 --dataset student
python main.py --gpu 0 --evaluate_type 2 --dataset obesity
python main.py --gpu 0 --evaluate_type 2 --dataset adult

// ******* OPRS ******* //
python main.py --gpu 0 --evaluate_type 3 --dataset student --mode 0 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 3 --dataset obesity --mode 0 --model MLP --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 3 --dataset adult --mode 0 --model MLP --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 3 --dataset student --mode 0 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 3 --dataset obesity --mode 0 --model ResNet --train_target --train_shadow
python main.py --gpu 0 --evaluate_type 3 --dataset adult --mode 0 --model ResNet --train_target --train_shadow

python main.py --gpu 0 --evaluate_type 3 --dataset student --mode 0 --model MLP
python main.py --gpu 0 --evaluate_type 3 --dataset obesity --mode 0 --model MLP 
python main.py --gpu 0 --evaluate_type 3 --dataset adult --mode 0 --model MLP 