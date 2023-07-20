#!/bin/bash -l
#SBATCH --job-name=captum
#SBATCH --time=0:15:00
#SBATCH -p v100
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=6               # total number of tasks across all nodes
#SBATCH --output=captum-%j.out
#SBATCH --error=captum-%j.err

pwd; hostname; date
echo jobid="${SLURM_JOB_ID}"; echo nodelist="${SLURM_JOB_NODELIST}"

module load python3/3.8.3_anaconda2020.07_mamba
# shellcheck disable=SC2006
__conda_setup="$(`which conda` 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

echo CUDA_VISIBLE_DEVICES: "$CUDA_VISIBLE_DEVICES"

cd /home/miran045/reine097/projects/loes-scoring-explainability || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/loes-scoring-explainability/src"
/home/miran045/reine097/projects/loes-scoring-explainability/venv/bin/python /home/miran045/reine097/projects/loes-scoring-explainability/src/my_captum_test.py
