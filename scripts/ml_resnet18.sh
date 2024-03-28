for trial in 1
do

    python ../src/main_ml.py --trial=$trial \
    --rounds=10 \
    --local_bs=32 \
    --lr=0.001 \
    --momentum=0.5 \
    --model=resnet18 \
    --size=224 \
    --gpu=1 \
    --seed=42

done 
