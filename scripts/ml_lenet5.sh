for trial in 1
do

    python ../src/main_ml.py --trial=$trial \
    --rounds=50 \
    --local_bs=128 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=lenet5 \
    --size=32 \
    --gpu=0 \
    --seed=42

done 
