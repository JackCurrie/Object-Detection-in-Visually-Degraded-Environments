for i in {1..500}
do
  python3 VGG19_transfer_frozen_hps.py with search_space -m hyper_param_search
done
