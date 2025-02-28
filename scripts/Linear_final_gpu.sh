if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Linear" ]; then
    mkdir ./logs/LongForecasting/Linear
fi

# For big datasets, memory costs are too big for parallel cpu training, and interative training is faster with gpu than with cpu

# Random seeds (10)
# seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
seeds=(10458 15227 3293 12890 15726 28649 25565 3144 32598 15349)

# For large datasets
# seeds_1=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
# seeds_1=(10458 15227 3293 12890 15726)
# seeds_2=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
# seeds_2=(28649 25565 3144 32598 15349)

model_names="Linear DLinear NLinear"


# for model_name in $model_names
# do
#    ###### Dataset: weather 
#    ###### Optimized
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset weather with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path weather.csv \
#             --model_id weather_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data custom \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 21 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size 16 --learning_rate 0.005 --individual >logs/LongForecasting/Linear/$model_name'_I_'weather_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset weather finished."

# for model_name in $model_names
# do
#    ###### Dataset: exchange_rate 
#    ###### Optimized
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       lr=0.0005
#       for seed in ${seeds[@]} 
#       do
#          bs=8
#          if [[ $pred_len -gt 300 ]]
#          then
#             bs=32
#          fi
#          echo "Training $model_name on dataset exchange-rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path exchange_rate.csv \
#             --model_id Exchange_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data custom \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 8 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/Linear/$model_name'_I_'exchange_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset exchange-rate finished."

# for model_name in $model_names
# do
#    ###### Dataset: ETTh1 
#    ###### Optimized
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset ETTh1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path ETTh1.csv \
#             --model_id ETTh1_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data ETTh1 \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 7 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size 32 --learning_rate 0.005 --individual >logs/LongForecasting/Linear/$model_name'_I_'ETTh1_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset ETTh1 finished."

# for model_name in $model_names
# do
#    ###### Dataset: ETTh2 
#    ###### Optimized (underperforming with respect to results from orignal work)
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       lr=0.05
#       bs=32
#       if [[ $model_name -eq "DLinear" ]]
#       then
#          bs=16
#          if [[ $pred_len -ge 336 ]]
#          then
#             lr=0.005
#             bs=32
#          fi
#       fi
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset ETTh2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path ETTh2.csv \
#             --model_id ETTh2_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data ETTh2 \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 7 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/Linear/$model_name'_I_'ETTh2_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset ETTh2 finished."

# for model_name in $model_names
# do
#    ###### Dataset: ETTm1
#    ###### Optimized
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset ETTm1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path ETTm1.csv \
#             --model_id ETTm1_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data ETTm1 \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 7 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size 8 --learning_rate 0.005 --individual >logs/LongForecasting/Linear/$model_name'_I_'ETTm1_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset ETTm1 finished."

# for model_name in $model_names
# do
#    ###### Dataset: ETTm2 
#    ###### Optimized (underperforming with respect to results from orignal work)
#    for pred_len in 96 192 336 720
#    do
#       seq_len=336
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset ETTm2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path ETTm2.csv \
#             --model_id ETTm2_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data ETTm2 \
#             --features M \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in 7 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size 32 --learning_rate 0.01 --individual >logs/LongForecasting/Linear/$model_name'_I_'ETTm2_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset ETTm2 finished."

# for model_name in $model_names
# do
#    ###### Dataset: national_illness 
#    ###### Optimized (slightly underperforming with respect to results from orignal work)
#    for pred_len in 24 36 48 60
#    do
#       seq_len=104
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset national illness (ILI) with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path national_illness.csv \
#             --model_id national_illness_$seq_len'_'$pred_len \
#             --model $model_name \
#             --data custom \
#             --features M \
#             --seq_len $seq_len \
#             --label_len 18 \
#             --pred_len $pred_len \
#             --enc_in 7 \
#             --train_epochs 20 --patience 5 \
#             --des 'Exp' \
#             --itr 1 --batch_size 32 --learning_rate 0.05 --individual >logs/LongForecasting/Linear/$model_name'_I_'ILI_$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       # wait
#       # echo "Done."
#    done
# done
# echo "Training of $model_name models on dataset national illness (ILI)) finished."

for model_name in $model_names
do
   ###### Dataset: electricity
   ###### Optimized
   for pred_len in 96 192 336 720
   # for pred_len in 96 720
   do
      seq_len=336
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset electricity with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id Electricity_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 321 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 16  --learning_rate 0.005 --individual >logs/LongForecasting/Linear/$model_name'_I_'electricity_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done
done
echo "Training of $model_name models on dataset electricity finished."

for model_name in $model_names
do
   ###### Dataset: traffic 
   ###### Optimized
   for pred_len in 96 192 336 720
   do
      seq_len=336
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset traffic with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path traffic.csv \
            --model_id traffic_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 862 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 16 --learning_rate 0.01 --individual >logs/LongForecasting/Linear/$model_name'_I_'traffic_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done
done
echo "Training of $model_name models on dataset traffic finished."

echo "Experiments finished"