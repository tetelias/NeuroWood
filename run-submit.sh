python src/preprocessing.py

nosetests

if [ ! -d models/ ]; then
  mkdir -p models/;
fi

wget https://www.dropbox.com/s/mczaqi2ox3kp2ho/timber_models.zip -P models/
unzip models/timber_models.zip -d models/
rm models/timber_models.zip

python src/training.py --gpu 0 --predict --test-fldr test

# python src/training.py --gpu 0 --predict --test-fldr test_top_scores