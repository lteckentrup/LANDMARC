#!/bin/bash
### Pair projection and historical 
declare -A hist_map=( 
    ["r1i1p1f1"]="a3bh_r1i1p1f1" 
    ["r2i1p1f1"]="a3o0_r6i1p1f1" 
    ["r3i1p1f1"]="a3nm_r7i1p1f1" 
    ["r4i1p1f1"]="a3o0_r6i1p1f1")

for exp_id in a7cy a7en a7eo; do
    echo "${exp_id}"
    for realisation_projection in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
        ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
        if [[ "${exp_id}" == "a7en" && "${realisation_projection}" == "r2i1p1f1" ]]; then
            realisation_projection="r4i1p1f1"
        fi
        
        ### Get corresponding historical experiment and realization
        hist_info=${hist_map[$realisation_projection]}
        
        ### Get exp_id for historical
        exp_hist=${hist_info%_*}  

        ### Get realisation for historical
        realisation_historical=${hist_info#*_}

        ### Make sure pairing worked
        echo "${realisation_projection}"
        echo ${exp_hist}
        echo ${realisation_historical}

        for var in tas pr rsds rlus rsus hfls; do
            echo ${var}
            ### Set up correct pathways
            ### Historical - esarchive
            pathwayIN_hist="/esarchive/exp/ecearth/${exp_hist}/original_files/cmorfiles/CMIP/EC-Earth-Consortium/"
            if [[ "${exp_hist}" == "a3bh" ]]; then
                pathwayIN_hist+="EC-Earth3-CC/historical/${realisation_historical}/Amon/${var}/gr/v20201127"
            else 
                pathwayIN_hist+="EC-Earth3-CC/historical/${realisation_historical}/Amon/${var}/gr/v????????"
            fi

            ### Projection - esarchive
            pathwayIN_projection="/esarchive/exp/ecearth/${exp_id}/original_files/cmorfiles/ScenarioMIP/"
            pathwayIN_projection+="EC-Earth-Consortium/EC-Earth3-CC/ssp245/${realisation_projection}/Amon/${var}/gr/v????????"

            ### Merged files /gpfs/scratch
            pathwayOUT="/gpfs/scratch/bsc32/bsc032352/LANDMARC/${exp_id}_${realisation_projection}/climate/${var}"

            #### Mask set negative precipitation to 0 
            if [ "$var" == "pr" ]; then
                cdo -L -mergetime \
                    ${pathwayIN_hist}/*nc ${pathwayIN_projection}/*nc \
                    ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012_tmp.nc
                cdo -L -setrtoc,-1000000,0,0 -muldpm -mulc,86400 \
                    ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012_tmp.nc \
                    ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012.nc
                rm ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012_tmp.nc
            else                                                           
                cdo mergetime ${pathwayIN_hist}/*nc ${pathwayIN_projection}/*nc \
                              ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012.nc
            fi

            ### Mask ocean pixels
            cdo div ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012.nc \
                    /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                    ${pathwayOUT}/${var}_Amon_EC-Earth3-CC_ssp245_${realisation_projection}_gr_185001-210012_land.nc            
        done
        for var in tasmax pr; do
            echo ${var}
            ### Set up correct pathways
            ### Historical - esarchive
            pathwayIN_hist="/esarchive/exp/ecearth/${exp_hist}/original_files/cmorfiles/CMIP/EC-Earth-Consortium/"
            if [[ "${exp_hist}" == "a3bh" ]]; then
                pathwayIN_hist+="EC-Earth3-CC/historical/${realisation_historical}/day/${var}/gr/v20210113"
            elif [[ "${exp_hist}" == "a3o0" ]]; then
                pathwayIN_hist="/gpfs/scratch/bsc32/bsc032352/LANDMARC/a3o0_r6i1p1f1/climate/"${var}
            else 
                pathwayIN_hist+="EC-Earth3-CC/historical/${realisation_historical}/day/${var}/gr/v????????"
            fi
            ### Projection - esarchive
            pathwayIN_projection="/esarchive/exp/ecearth/${exp_id}/original_files/cmorfiles/ScenarioMIP/"
            pathwayIN_projection+="EC-Earth-Consortium/EC-Earth3-CC/ssp245/${realisation_projection}/day/${var}/gr/v????????"

            ### Merged files /gpfs/scratch
            pathwayOUT="/gpfs/scratch/bsc32/bsc032352/LANDMARC/${exp_id}_${realisation_projection}/climate/${var}"

            #### Mask set negative precipitation to 0 
            if [ "$var" == "pr" ]; then
                for year in {1850..2014..1}; do
                    echo ${year}
                    cdo -L -mulc,86400 -setrtoc,-1000000,0,0 \
                        ${pathwayIN_hist}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231_tmp.nc
                    cdo div \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231_tmp.nc \
                        /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231.nc
                    rm ${pathwayOUT}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231_tmp.nc
                done
                for year in {2015..2100..1}; do
                    echo ${year}
                    cdo -L -mulc,86400 -setrtoc,-1000000,0,0 \
                        ${pathwayIN_projection}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231_tmp.nc
                    cdo div \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231_tmp.nc \
                        /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231.nc
                    rm ${pathwayOUT}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231_tmp.nc
                done
            else                                                           
                for year in {1850..2014..1}; do
                    echo ${year}
                    cdo div \
                        ${pathwayIN_hist}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231.nc \
                        /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_historical_${realisation_historical}_gr_${year}0101-${year}1231.nc
                done
                for year in {2015..2100..1}; do
                    echo ${year}
                    cdo div \
                        ${pathwayIN_projection}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231.nc \
                        /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                        ${pathwayOUT}/${var}_day_EC-Earth3-CC_ssp245_${realisation_projection}_gr_${year}0101-${year}1231.nc
                done
            fi
        done
    done
done

### Calculate albedo
for exp_id in a7cy a7en a7eo; do
    echo ${exp_id}
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    for realisation in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
        if [[ "${exp_id}" == "a7en" && "${realisation}" == "r2i1p1f1" ]]; then
            realisation="r4i1p1f1"
        fi
        echo ${realisation}
        pathwayIN="/gpfs/scratch/bsc32/bsc032352/LANDMARC/${exp_id}_${realisation}/climate"
        cdo -L -chname,rsus,albedo -div \
                ${pathwayIN}/rsus/rsus_Amon_EC-Earth3-CC_ssp245_${realisation}_gr_185001-210012.nc \
                ${pathwayIN}/rsds/rsds_Amon_EC-Earth3-CC_ssp245_${realisation}_gr_185001-210012.nc \
                ${pathwayIN}/albedo/albedo_Amon_EC-Earth3-CC_ssp245_${realisation}_gr_185001-210012.nc
        cdo div ${pathwayIN}/albedo/albedo_Amon_EC-Earth3-CC_ssp245_${realisation}_gr_185001-210012.nc \
                /gpfs/scratch/bsc32/bsc032352/LANDMARC/aux/landmask.nc \
                 ${pathwayIN}/albedo/albedo_Amon_EC-Earth3-CC_ssp245_${realisation}_gr_185001-210012_land.nc
    done
done
