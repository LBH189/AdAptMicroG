python AdAptMicroG.py -i Input_files/CRC_10fold.csv -t 1  -d CRC -o AdAptMicroG_new_pipe/CRC_10fold

python AdAptMicroG.py -i Input_files/Obt_10fold.csv -t 1  -d Obt -o AdAptMicroG_new_pipe/Obt_10fold

python AdAptMicroG.py -i Input_files/T2D_10fold.csv -t 1  -d T2D -o AdAptMicroG_new_pipe/T2D_10fold

python AdAptMicroG.py -i Input_files/ASD_10fold.csv -t 1  -d ASD -o AdAptMicroG_new_pipe/ASD_10fold

python AdAptMicroG.py -i Input_files/MHE_10fold.csv -t 1  -d MHE -o AdAptMicroG_new_pipe/MHE_10fold

python AdAptMicroG.py -i Input_files/cirrhosis_10fold.csv -t 1  -d Cirrhosis -o AdAptMicroG_new_pipe/cirrhosis_10fold


python AdAptMicroG.py -i Input_files/CRC_FRA_LOSO.csv -s 10  -d CRC -o AdAptMicroG_new_pipe/CRC_FRA_LOSO

python AdAptMicroG.py -i Input_files/CRC_GER_LOSO.csv -s 10  -d CRC -o AdAptMicroG_new_pipe/CRC_GER_LOSO

python AdAptMicroG.py -i Input_files/CRC_USA_LOSO.csv -s 10  -d CRC -o AdAptMicroG_new_pipe/CRC_USA_LOSO

python AdAptMicroG.py -i Input_files/IBD_UK_LOSO.csv -s 10  -d IBD -o AdAptMicroG_new_pipe/IBD_UK_LOSO

python AdAptMicroG.py -i Input_files/IBD_DK_LOSO.csv -s 10  -d IBD -o AdAptMicroG_new_pipe/IBD_DK_LOSO

python AdAptMicroG.py -i Input_files/CRC_IND_additional.csv -s 10 -b 32 -d CRC -o AdAptMicroG_new_pipe/CRC_IND_Additional
