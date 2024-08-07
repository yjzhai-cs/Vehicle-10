for trial in 1
do

    python ../src/main_ml.py --trial=$trial \
    --rounds=50 \
    --local_bs=128 \
    --lr=0.001 \
    --momentum=0.9 \
    --model=resnet34 \
    --size=224 \
    --gpu=2 \
    --seed=42

done 
