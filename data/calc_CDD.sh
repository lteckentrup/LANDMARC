pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC'
realizations=(r1i1p1f1 r2i1p1f1 r3i1p1f1)
experiments=(a7cy a7en a7eo)

for exp_id in "${experiments[@]}"; do
    echo ${exp_id}
    for realisation in "${realizations[@]}"; do
        if [[ "${exp_id}" == "a7en" && "${realisation}" == "r2i1p1f1" ]]; then
            realisation="r4i1p1f1"
        fi  
        echo ${realisation}

        ### Make directory
        mkdir ${pathwayIN}/${exp_id}_${realisation}/climate/CDD

        ### Separate into years
        cdo splityear \
            ${pathwayIN}/${exp_id}_${realisation}/climate/pr/pr_day_EC-Earth3-CC_ssp245_${realisation}_gr_18500101-21001231.nc \
            ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/pr_day_EC-Earth3-CC_ssp245_${realisation}_gr_
        
        ### Loop through all years and calculate CDD
        for year in {1850..2100..1}; do
            echo ${year}
            cdo -L -chname,consecutive_dry_days_index_per_time_period,CDD -selname,consecutive_dry_days_index_per_time_period -eca_cdd \
                ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/pr_day_EC-Earth3-CC_ssp245_${realisation}_gr_${year}.nc \
                ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/CDD_day_EC-Earth3-CC_ssp245_${realisation}_gr_${year}.nc
        done

        ### Merge files
        cdo mergetime \
            ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/CDD_day_EC-Earth3-CC_ssp245_${realisation}_gr_????.nc \
            ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/CDD_day_EC-Earth3-CC_ssp245_${realisation}_gr_1850-2100.nc
        
        for year in {1850..2100..1}; do
            rm ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/pr_day_EC-Earth3-CC_ssp245_${realisation}_gr_${year}.nc
            rm ${pathwayIN}/${exp_id}_${realisation}/climate/CDD/CDD_day_EC-Earth3-CC_ssp245_${realisation}_gr_${year}.nc
        done
    done
done
