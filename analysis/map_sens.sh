for region in global ACTO ASEAN EAC EU27 NAMERICA OCEANIA; do
    echo ${region}
    python maps_sens.py --region ${region} --first_year '2070' --last_year '2099' --exp 'a6xx' --ref 'a6xv' --plot_type 'cpool'
    python maps_sens.py --region ${region} --first_year '2070' --last_year '2099' --plot_type 'clim'
    for exp in a6xx a6xv; do
        echo ${exp}
        python maps_sens.py --region ${region} --first_year '2070' --last_year '2099' --exp ${exp} --ref 'a6zt' --plot_type 'cpool'
    done
done   
