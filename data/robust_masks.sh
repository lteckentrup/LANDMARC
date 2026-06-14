#!/bin/bash

# SLURM configuration for MN5
#SBATCH --output=robust_masks_%j.out
#SBATCH --error=robust_masks_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=gp_debug
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=lina.teckentrup@bsc.es
#sbatch -A bsc32 -q gp_bsces robust_masks.sh

# Load appropriate modules and activate conda environment
if [ "$BSC_MACHINE" = "mn5" ]; then

    export MODULEPATH=$MODULEPATH:/apps/GPP/EASYBUILD/modules/all
    source /apps/GPP/MINICONDA/24.1.2/etc/profile.d/conda.sh
    conda activate joni

elif [ "$BSC_MACHINE" = "nord3v2" ]; then
    module load GDAL/3.3.2-foss-2019b-Python-3.7.4
    module load pycdo/1.5.3-foss-2019b-Python-3.7.4
    module load pynco/0.0.2-foss-2019b-Python-3.7.4
fi


python robust_masks.py --var albedo --exp_ref a7cy --exp_pert a7cy --first_year 2071 --last_year 2100
python robust_masks.py --var albedo --exp_ref a7cy --exp_pert a7en --first_year 2071 --last_year 2100
python robust_masks.py --var albedo --exp_ref a7en --exp_pert a7eo --first_year 2071 --last_year 2100

python robust_masks.py --var hfls --exp_ref a7cy --exp_pert a7cy --first_year 2071 --last_year 2100
python robust_masks.py --var hfls --exp_ref a7cy --exp_pert a7en --first_year 2071 --last_year 2100
python robust_masks.py --var hfls --exp_ref a7en --exp_pert a7eo --first_year 2071 --last_year 2100

python robust_masks.py --var cLand --exp_ref a7cy --exp_pert a7cy --first_year 2071 --last_year 2100
python robust_masks.py --var cLand --exp_ref a7cy --exp_pert a7en --first_year 2071 --last_year 2100
python robust_masks.py --var cLand --exp_ref a7en --exp_pert a7eo --first_year 2071 --last_year 2100
