#!/bin/sh

#SBATCH --job-name=loes-scoring-alex-net # job name

#SBATCH --mem=180g        # memory per cpu-core (what is the default?)
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-alex-net-%j.err
#SBATCH -o loes-scoring-alex-net-%j.out

#SBATCH -A feczk001

cd /home/miran045/reine097/projects/MyCaptum/ || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/MyCaptum/src"
/home/miran045/reine097/projects/MyCaptum/venv/bin/python \
  /home/miran045/reine097/projects/MyCaptum/src/dcan/training/training.py \
  --csv-data-file "/home/miran045/reine097/projects/loes-scoring-2/data/MNI-space_Loes_data.csv" \
  --anatomical-region all --batch-size 1 --num-workers 1 --epochs 4 \
  --model-save-location "/home/miran045/reine097/projects/MyCaptum/models/loes_scoring_03.pt" --use-gd-only 1
