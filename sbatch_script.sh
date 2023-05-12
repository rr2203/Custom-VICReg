
#!/bin/bash -e
#SBATCH --output=%x_%j.txt --time=18:00:00 --wrap "sleep infinity" --gres=gpu:1
echo "Hostname: $(hostname)"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $4}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi | grep MiB |  awk '{print $9 $10 $11}')"
module purge
module load intel/19.1.2
module load python/intel/3.8.6
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd /scratch/sj4020/rahul-stuff
venv_name="vicreg_training_venv"
if [ ! -d "$venv_name" ]; then
  python -m venv "$venv_name"
fi
source ./$venv_name/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install  --use-feature=2020-resolver  torchvision opencv-python opencv-python matplotlib 'git+https://github.com/facebookresearch/segment-anything.git'



pip3 install torch torchvision
pip3 install submitit
python3 run_with_submitit.py --nodes 4 --ngpus 8 --data-dir /scratch/sj4020/vicreg/imagenet --exp-dir /scratch/sj4020/vicreg/experiment --arch resnet50 --epochs 1000 --batch-size 2048 --base-lr 0.2
