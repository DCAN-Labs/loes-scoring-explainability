Saliency mapping of Loes score models
=====================================

* Location of original MRIs
* Location of saliency maps
  * /home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data_saliency


---

```
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

#SBATCH -A miran045

cd /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/loes-scoring-2/src:/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training/training.py \
  --csv-data-file "/home/miran045/reine097/projects/loes-scoring-2/data/filtered/ashish_all.csv" \
  --batch-size 1 --num-workers 1 --epochs 128 \
  --model-save-location "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_01_temp.pt" \
  --plot-location "/home/miran045/reine097/projects/loes-scoring-2/doc/img/model01_temp.png"
```

[Model 01](/home/miran045/reine097/projects/loes-scoring-2/doc/img/model01_temp.png)
Low Loes score example
----------------------

`/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-7151DOBU_ses-20170303_space-MNI_mprage.nii.gz,,0,1.0,igor`

```
module load fsl
fsleyes
```

Middle Loes score example
---------------------------------------

High Loes score example
-----------------------

`/home/feczk001/shared/projects/S1067_Loes/data/MNI-space_Loes_data/sub-6630SICH_ses-20160727_space-MNI_mprageGd.nii.gz,,1,21.0`

