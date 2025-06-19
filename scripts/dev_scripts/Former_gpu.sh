if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Transformers" ]; then
    mkdir ./logs/LongForecasting/Transformers
fi

# Same seeds as Linear models
seeds=(32598)

model_names="Autoformer"

for model_name in $model_names
do
   pred_lengths=(192)

   # ###### Dataset: electricity
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]} 
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset electricity with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path electricity.csv \
   #          --model_id electricity_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data custom \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 321 \
   #          --dec_in 321 \
   #          --c_out 321 \
   #          --des 'Exp' \
   #          --itr 1 >logs/LongForecasting/Transformers/$model_name'_electricity_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done


   # ###### Dataset: traffic 
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset traffic with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path traffic.csv \
   #          --model_id traffic_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data custom \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 862 \
   #          --dec_in 862 \
   #          --c_out 862 \
   #          --des 'Exp' \
   #          --itr 1 \
   #          --train_epochs 10 >logs/LongForecasting/Transformers/$model_name'_traffic_'$seq_len'_'$pred_len'_seed_'$seed.log
   #          # --train_epochs 3 >logs/LongForecasting/Transformers/$model_name'_traffic_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done

   # ###### Dataset: weather 
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset weather with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path weather.csv \
   #          --model_id weather_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data custom \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 21 \
   #          --dec_in 21 \
   #          --c_out 21 \
   #          --des 'Exp' \
   #          --itr 1 \
   #          --train_epochs 10 >logs/LongForecasting/Transformers/$model_name'_weather_'$seq_len'_'$pred_len'_seed_'$seed.log
   #          # --train_epochs 2 >logs/LongForecasting/Transformers/$model_name'_weather_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done

   # ###### Dataset: exchange_rate 
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset exchange rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path exchange_rate.csv \
   #          --model_id exchange_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data custom \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 8 \
   #          --dec_in 8 \
   #          --c_out 8 \
   #          --des 'Exp' \
   #          --itr 1 \
   #          --train_epochs 10 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
   #          # --train_epochs 1 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done


   # ###### Dataset: ETTh1 
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset ETTh1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path ETTh1.csv \
   #          --model_id ETTh1_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data ETTh1 \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 7 \
   #          --dec_in 7 \
   #          --c_out 7 \
   #          --des 'Exp' \
   #          --itr 1  >logs/LongForecasting/Transformers/$model_name'_ETTh1_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done


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
            --save_pred_values True \
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
            --itr 1  >logs/LongForecasting/Transformers/$model_name'_ETTh2_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
      echo "Done."
   done


   # ###### Dataset: ETTm1
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset ETTm1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path ETTm1.csv \
   #          --model_id ETTm1_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data ETTm1 \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 7 \
   #          --dec_in 7 \
   #          --c_out 7 \
   #          --des 'Exp' \
   #          --itr 1  >logs/LongForecasting/Transformers/$model_name'_ETTm1_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done


   # ###### Dataset: ETTm2 
   # ###### Optimized
   # for pred_len in ${pred_lengths[@]}
   # do
   #    seq_len=96
   #    for seed in ${seeds[@]} 
   #    do
   #       echo "Training $model_name on dataset ETTm2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
   #       python -u run_longExp.py \
   #          --is_training 1 \
   #          --seed $seed \
   #          --num_workers 0 \
   #          --root_path ./dataset/ \
   #          --data_path ETTm2.csv \
   #          --model_id ETTm2_$seq_len_$pred_len \
   #          --model $model_name \
   #          --data ETTm2 \
   #          --features M \
   #          --seq_len $seq_len \
   #          --label_len 48 \
   #          --pred_len $pred_len \
   #          --e_layers 2 \
   #          --d_layers 1 \
   #          --factor 3 \
   #          --enc_in 7 \
   #          --dec_in 7 \
   #          --c_out 7 \
   #          --des 'Exp' \
   #          --itr 1  >logs/LongForecasting/Transformers/$model_name'_ETTm2_'$seq_len'_'$pred_len'_seed_'$seed.log
   #    done
   #    echo "Done."
   # done

# pred_lengths=(24 36 48 60)

#    ###### Dataset: national_illness 
#    ###### Optimized
#    for pred_len in ${pred_lengths[@]}
#    do
#       seq_len=36
#       for seed in ${seeds[@]} 
#       do
#          echo "Training $model_name on dataset national illness (ILI) with seq_len $seq_len, pred_len $pred_len and seed $seed..."
#          python -u run_longExp.py \
#             --is_training 1 \
#             --seed $seed \
#             --num_workers 0 \
#             --root_path ./dataset/ \
#             --data_path national_illness.csv \
#             --model_id ili_$seq_len_$pred_len \
#             --model $model_name \
#             --data custom \
#             --features M \
#             --seq_len $seq_len \
#             --label_len 18 \
#             --pred_len $pred_len \
#             --e_layers 2 \
#             --d_layers 1 \
#             --factor 3 \
#             --enc_in 7 \
#             --dec_in 7 \
#             --c_out 7 \
#             --des 'Exp' \
#             --itr 1 >logs/LongForecasting/Transformers/$model_name'_ILI_'$seq_len'_'$pred_len'_seed_'$seed.log
#       done
#       echo "Done."
#    done
done