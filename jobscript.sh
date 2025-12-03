#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J deep-learning-gnn-experiments
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 11:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/job/gpu_%J.out
#BSUB -e logs/job/gpu_%J.err
# -- end of LSF options --

# Activate virtualenv
if [ -f .venv/bin/activate ]; then
	# shellcheck disable=SC1091
	. .venv/bin/activate
fi

python src/run.py +experiments=gin_best_1500
python src/run.py +experiments=graph_mixup/gin_best_1500_mixup