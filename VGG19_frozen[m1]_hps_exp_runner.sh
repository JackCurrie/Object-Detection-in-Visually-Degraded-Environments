for i in {1..150}
do
  python3 VGG19_transfer_frozen[m1]_hps.py with search_space -m hps_m1_aug
done
