for trial in 1
do

    python ../src/main_ml.py --trial=$trial \
    --rounds=50 \
    --local_bs=128 \
    --lr=0.001 \
    --momentum=0.5 \
    --model=resnet9 \
    --size=32 \
    --gpu=1 \
    --seed=42

done 
