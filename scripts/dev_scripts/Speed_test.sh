if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Speed" ]; then
    mkdir ./logs/LongForecasting/Speed
fi

seeds=(0)

model_names="Autoformer Informer Transformer"
# model_names="Transformer"

for model_name in $model_names
do
   pred_lengths=(96 720)

   ###### Dataset: electricity
   ###### Optimized
   for pred_len in ${pred_lengths[@]} 
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset electricity with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id electricity_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des 'Exp' \
            --itr 1 >logs/LongForecasting/Speed/$model_name'_electricity_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done

   ###### Dataset: traffic 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset traffic with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path traffic.csv \
            --model_id traffic_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des 'Exp' \
            --itr 1 \
            --train_epochs 10 >logs/LongForecasting/Speed/$model_name'_traffic_'$seq_len'_'$pred_len'_seed_'$seed.log
            # --train_epochs 3 >logs/LongForecasting/Transformers/$model_name'_traffic_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done

   ###### Dataset: weather 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset weather with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path weather.csv \
            --model_id weather_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des 'Exp' \
            --itr 1 \
            --train_epochs 10 >logs/LongForecasting/Speed/$model_name'_weather_'$seq_len'_'$pred_len'_seed_'$seed.log
            # --train_epochs 2 >logs/LongForecasting/Transformers/$model_name'_weather_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done

   ###### Dataset: exchange_rate 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset exchange rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path exchange_rate.csv \
            --model_id exchange_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 8 \
            --dec_in 8 \
            --c_out 8 \
            --des 'Exp' \
            --itr 1 \
            --train_epochs 10 >logs/LongForecasting/Speed/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
            # --train_epochs 1 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done


   ###### Dataset: ETTh1 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTh1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_$seq_len_$pred_len \
            --model $model_name \
            --data ETTh1 \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1  >logs/LongForecasting/Speed/$model_name'_ETTh1_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done


   ###### Dataset: ETTh2 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTh2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh2.csv \
            --model_id ETTh2_$seq_len_$pred_len \
            --model $model_name \
            --data ETTh2 \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1  >logs/LongForecasting/Speed/$model_name'_ETTh2_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done


   ###### Dataset: ETTm1
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTm1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTm1.csv \
            --model_id ETTm1_$seq_len_$pred_len \
            --model $model_name \
            --data ETTm1 \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1  >logs/LongForecasting/Speed/$model_name'_ETTm1_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done


   ###### Dataset: ETTm2 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTm2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTm2.csv \
            --model_id ETTm2_$seq_len_$pred_len \
            --model $model_name \
            --data ETTm2 \
            --features M \
            --seq_len $seq_len \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1  >logs/LongForecasting/Speed/$model_name'_ETTm2_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done

   pred_lengths=(24 60)

   ###### Dataset: national_illness 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=36
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset national illness (ILI) with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path national_illness.csv \
            --model_id ili_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 18 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1 >logs/LongForecasting/Speed/$model_name'_ILI_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
   done
done


# ###################################################################
model_names="Linear DLinear NLinear"

for model_name in $model_names
do
   ###### Dataset: weather 
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset weather with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path weather.csv \
            --model_id weather_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 21 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 16 --learning_rate 0.005 --individual >logs/LongForecasting/Speed/$model_name'_I_'weather_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: exchange_rate 
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
      lr=0.0005
      for seed in ${seeds[@]} 
      do
         bs=8
         if [[ $pred_len -gt 300 ]]
         then
            bs=32
         fi
         echo "Training $model_name on dataset exchange-rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path exchange_rate.csv \
            --model_id Exchange_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 8 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/Speed/$model_name'_I_'exchange_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: ETTh1 
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTh1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTh1 \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --learning_rate 0.005 --individual >logs/LongForecasting/Speed/$model_name'_I_'ETTh1_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: ETTh2 
   ###### Optimized (underperforming with respect to results from orignal work)
   for pred_len in 96 720
   do
      seq_len=96
      lr=0.05
      bs=32
      if [[ $model_name -eq "DLinear" ]]
      then
         bs=16
         if [[ $pred_len -ge 336 ]]
         then
            lr=0.005
            bs=32
         fi
      fi
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTh2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh2.csv \
            --model_id ETTh2_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTh2 \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/Speed/$model_name'_I_'ETTh2_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: ETTm1
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTm1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTm1.csv \
            --model_id ETTm1_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTm1 \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 8 --learning_rate 0.005 --individual >logs/LongForecasting/Speed/$model_name'_I_'ETTm1_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: ETTm2 
   ###### Optimized (underperforming with respect to results from orignal work)
   for pred_len in 96 720
   do
      seq_len=96
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTm2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTm2.csv \
            --model_id ETTm2_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTm2 \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --learning_rate 0.01 --individual >logs/LongForecasting/Speed/$model_name'_I_'ETTm2_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: national_illness 
   ###### Optimized (slightly underperforming with respect to results from orignal work)
   for pred_len in 24 60
   do
      seq_len=36
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset national illness (ILI) with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path national_illness.csv \
            --model_id national_illness_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len 18 \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 32 --learning_rate 0.05 --individual >logs/LongForecasting/Speed/$model_name'_I_'ILI_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: electricity
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
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
            --itr 1 --batch_size 16  --learning_rate 0.005 --individual >logs/LongForecasting/Speed/$model_name'_I_'electricity_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done


   ###### Dataset: traffic 
   ###### Optimized
   for pred_len in 96 720
   do
      seq_len=96
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
            --itr 1 --batch_size 16 --learning_rate 0.01 --individual >logs/LongForecasting/Speed/$model_name'_I_'traffic_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done
done

echo "Experiments finished"