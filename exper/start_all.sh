#!/bin/bash


for sim in "1" "2" "3" "4" "6"
do
  cd sim${sim}
  source start_sbatch.sh
  cd ..
done 

cd sim5
sbatch sbatch_sim5
cd ..
