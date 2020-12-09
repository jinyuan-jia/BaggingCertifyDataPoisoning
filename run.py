
import os 


os.system("nohup python3 -u training_mnist_bagging.py --k 30 --end 1000 --gpu 0 --gpum 0.05 > ./release_30.log &")
os.system("nohup python3 -u training_mnist_bagging.py --k 50 --end 1000 --gpu 1 --gpum 0.05 > ./release_50.log &")
os.system("nohup python3 -u training_mnist_bagging.py --k 100 --end 1000 --gpu 2 --gpum 0.05 > ./release_100.log &")

os.system("python3 compute_certified_poisoning_size.py --k 30")
os.system("python3 compute_certified_poisoning_size.py --k 50")
os.system("python3 compute_certified_poisoning_size.py --k 100")

