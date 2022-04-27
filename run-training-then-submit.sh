python src/preprocessing.py

nosetests

python src/training.py --gpu 0 --ep 11 --lr 1.2e-3 --fold 1
python src/training.py --gpu 0 --ep 11 --lr 1.2e-3 --fold 2
python src/training.py --gpu 0 --ep 15 --lr 1e-3   --fold 0
python src/training.py --gpu 0 --ep 15 --lr 1e-3   --fold 1
python src/training.py --gpu 0 --ep 15 --lr 1.2e-3 --fold 0
python src/training.py --gpu 0 --ep 15 --lr 1.2e-3 --fold 1
python src/training.py --gpu 0 --ep 15 --lr 1.2e-3 --fold 4
python src/training.py --gpu 0 --ep 16 --lr 0.8e-3 --fold 4
python src/training.py --gpu 0 --ep 16 --lr 1e-3   --fold 2 --use-fmix
python src/training.py --gpu 0 --ep 16 --lr 1e-3   --fold 3 --use-fmix
python src/training.py --gpu 0 --ep 16 --lr 1.1e-3 --fold 2 --use-fmix
python src/training.py --gpu 0 --ep 16 --lr 1.1e-3 --fold 3 --use-fmix

python src/training.py --gpu 0 --predict --test-fldr test

# python src/training.py --gpu 0 --predict --test-fldr test_top_scores