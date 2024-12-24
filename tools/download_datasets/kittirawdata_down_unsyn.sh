#!/bin/bash

files=(
2011_10_03_drive_0027
2011_10_03_drive_0042
2011_10_03_drive_0034
2011_09_26_drive_0067
2011_09_30_drive_0016
2011_09_30_drive_0018
2011_09_30_drive_0020
2011_09_30_drive_0027
2011_09_30_drive_0028
2011_09_30_drive_0033
2011_09_30_drive_0034
)

for i in ${files[@]}; do
        #if [ ${i:(-3)} != "zip" ]
        #then
       	        shortname=$i'_extract.zip'
                fullname=$i'/'$i'_extract.zip'
        #else
        #        shortname=$i
        #        fullname=$i
        #fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done

