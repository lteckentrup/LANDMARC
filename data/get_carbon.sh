#!/bin/bash
variables_Eyr=(cBECCS cBiochar fco2fossub fBECCS fBiochar)
variables_Emon=(cLand cSoil)
variables_Lmon=(cVeg cLitter)

flux_vars=(fLuc fHarvest nbp npp fFire)  # Variables that require yearsum conversion

fileOUT='/gpfs/scratch/bsc32/bsc032352/LANDMARC'
prefix='/esarchive/exp/ecearth'

#### HISTORICAL 
### Pre-process historical Emon and Lmon variables
for time_freq in "Emon" "Lmon"; do
    echo ${time_freq}
    exp_id=a766
    realisation=r1i1p1f1

    suffix="original_files/cmorfiles/ScenarioMIP/EC-Earth-Consortium/EC-Earth3-CC"
    suffix+="/ssp126/${realisation}/${time_freq}"

    if [[ "$time_freq" == "Emon" ]]; then
        variables=("${variables_Emon[@]}")
    else
        variables=("${variables_Lmon[@]}")
    fi

    for var in "${variables[@]}"; do
        echo ${var}   
        output_file="${fileOUT}/${exp_id}_${realisation}/carbon/${var}/"
        output_file+="${var}_${time_freq}_EC-Earth3-CC_historical_${realisation}_gr_1850-2014.nc"
        output_file_tmp="${output_file}.tmp"
        for flux_var in "${flux_vars[@]}"; do
            cdo -mergetime \
                ${prefix}/${exp_id}/${suffix}/${var}/*/*/* \
                "${output_file_tmp}"
            if [[ "$flux_var" == "$var" ]]; then
                cdo -L -yearsum -mulc,86400 -muldpm \
                    "${output_file_tmp}" \
                    "${output_file}"
            else
                cdo -L -divdpy -yearsum -muldpm \
                    $"${output_file_tmp}" \
                    "${output_file}"
            fi
            rm "${output_file_tmp}"
        done
    done
done

######### PROJECTIONS
realizations=(r1i1p1f1 r2i1p1f1 r3i1p1f1)
experiments=(a7cy a7en a7eo)

### Pre-process ssp245 Emon and Lmon variables
for time_freq in "Emon" "Eyr" "Lmon"; do
    echo ${time_freq}
    for exp_id in "${experiments[@]}"; do
        echo ${exp_id}
        ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
        for realisation in "${realizations[@]}"; do
            if [[ "${exp_id}" == "a7en" && "${realisation}" == "r2i1p1f1" ]]; then
                realisation="r4i1p1f1"
            fi
            echo ${realisation}
            suffix="original_files/cmorfiles/ScenarioMIP/EC-Earth-Consortium/EC-Earth3-CC"
            suffix+="/ssp245/${realisation}/${time_freq}"

            ### Select variables from TableID
            if [[ "$time_freq" == "Emon" ]]; then
                variables=("${variables_Emon}")
            elif [[ "$time_freq" == "Lmon" ]]; then
                variables=("${variables_Lmon}")
            else
                variables=("${variables_Eyr}")
            fi


            ### Loop through variables
            for var in "${variables[@]}"; do
                echo ${var}
                output_file="${fileOUT}/${exp_id}_${realisation}/carbon/${var}/"
                output_file+="${var}_${time_freq}_EC-Earth3-CC_ssp245_${realisation}_gr_2015-2100.nc"
                output_file_tmp="${fileOUT}/${exp_id}_${realisation}/carbon/${var}/"
                output_file_tmp+="${var}_${time_freq}_EC-Earth3-CC_ssp245_${realisation}_gr_2015-2100_tmp.nc"

                echo "${output_file}"
                echo "${output_file_tmp}"

                ### Only merge annual values and get annual sums from monthly outputs
                if [[ "$time_freq" == "Eyr" ]]; then
                    cdo -L -mergetime \
                        ${prefix}/${exp_id}/${suffix}/${var}/*/*/* \
                        "${output_file}"
                else
                    cdo -L -mergetime \
                        ${prefix}/${exp_id}/${suffix}/${var}/*/*/* \
                        "${output_file_tmp}"

                    if [[ " ${flux_vars[@]} " =~ " ${var} " ]]; then
                        cdo -L -yearsum -mulc,86400 -muldpm \
                            "${output_file_tmp}" \
                            "${output_file}"
                    else
                        cdo -L -divdpy -yearsum -muldpm \
                            "${output_file_tmp}" \
                            "${output_file}"
                    fi
                    rm "${output_file_tmp}"
                fi
            done
        done
    done
done
