for trial in 1
do

    python ../src/main_fedprox.py --trial=$trial \
    --rounds=300 \
    --local_ep=10 \
    --num_users=100 \
    --frac=0.2 \
    --local_bs=32 \
    --bs=128 \
    --lr=0.007 \
    --momentum=0.9 \
    --model=resnet9 \
    --size=32 \
    --gpu=4 \
    --seed=42 \
    --partition='percentage20' \
    --datadir='../data/' \
    --dataset=vehicle10 \
    --print_freq=10 \
    --local_view \
    --mu=0.001 \

done 
