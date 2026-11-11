
#!/bin/bash
set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

mkdir -p ./logs/APAVA
log_dir="./logs/APAVA"

model_name=REED
data_path="./dataset/APAVA/"
data_name="APAVA"

bss=(128)
lrs=(7e-5)
t_layers=(6)
v_layers=(6)
dropouts=(0.0)
d_models=(256)
patch_lens=(1)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for t_layer in "${t_layers[@]}"; do
            for v_layer in "${v_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                        for patch_len in "${patch_lens[@]}"; do

                            python -u run.py \
                                --model "$model_name" \
                                --dataset "$data_name" \
                                --data-dir "$data_path" \
                                --temporal-depth "$t_layer" \
                                --channel-depth "$v_layer" \
                                --batch-size "$bs" \
                                --d_model "$d_model" \
                                --dropout "$dropout" \
                                --patch_len "$patch_len" \
                                --augmentation-policy "flip0.2,frequency0.2,jitter0.,mask0.,channel0.,drop0.4" \
                                --enable-temporal-delta \
                                --enable-curvature-residual \
                                --curvature-scale 0.01 \
                                --enable-subspace-penalty \
                                --subspace-weight 0.005 \
                                --subspace-warmup 10 \
                                --lr-schedule constant \
                                --runs 5 \
                                --seed 42 \
                                --lr "$lr" \
                                --epochs 50 \
                                --patience 25 \
                                --gpu 0 \
                                > "${log_dir}/REED_bs${bs}_lr${lr}_tl${t_layer}_vl${v_layer}_dp${dropout}_dm${d_model}_pl${patch_len}.log" 2>&1

                        done
                    done
                done
            done
        done
    done
done

echo "REED APAVA experiments completed."
