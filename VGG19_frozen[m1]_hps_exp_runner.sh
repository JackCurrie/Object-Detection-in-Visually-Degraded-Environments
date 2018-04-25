for i in {1..200}
do
  python3 VGG19_transfer_frozen[m1]_hps.py with search_space -m hps_m1
done
