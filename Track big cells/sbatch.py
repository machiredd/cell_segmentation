#!/bin/sh

#SBATCH --job-name=mbo
#SBATCH --output=mask_bor_of.out
#SBATCH --time=1-00:00:00

module use /home/exacloud/software/modules
module load cudnn/7.6-10.1
module load cuda/10.1.243

eval "$(conda shell.bash hook)"
conda activate em3

python iou_parallel_big_cells.py --mask_dir /home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_big/data/new/mask_bor_of/ --gt_dir /home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_big/data/Cropped/ --dest_dir /home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_big/process/results/mask_bor_of_aligned
