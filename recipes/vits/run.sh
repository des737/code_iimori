#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
COMMON_ROOT=../common
. $COMMON_ROOT/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

stage=0
stop_stage=0
overwrite=0
local_dir=""

. $COMMON_ROOT/parse_options.sh || exit 1;

dumpdir=dump
dump_org_dir=$dumpdir/${spk}_sr${sample_rate}/org
dump_norm_dir=$dumpdir/${spk}_sr${sample_rate}/norm

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname
# configを残しておく.
mkdir -p $expdir
mkdir -p $expdir/conf
cp ./config.yaml $expdir


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for fastspeech2"

    if [ -e $dump_org_dir ] && [ ${overwrite} -le 0 ]; then
        echo "dump files are already exists. you cant run this preprocess."
        echo "if you want to overwrite, add --overwrite 1."
        exit 1
    fi

    xrun python emb_preprocess.py $dialogue_info \
        $dumpdir/${spk}_sr${sample_rate} --BERT_weight $BERT_weight \
        --input_duration_paths $input_duration_paths --output_mel_file_paths $output_mel_file_paths \
        --model_config_paths $model_config_paths --pretrained_checkpoints $pretrained_checkpoints \
        --input_wav_paths $input_wav_paths --input_lab_paths $input_lab_paths --input_textgrid_paths $input_textgrid_paths \
        --sample_rate $sample_rate --filter_length $filter_length \
        --hop_length $hop_length --win_length $win_length \
        --n_mel_channels $n_mel_channels --mel_fmin $mel_fmin --mel_fmax $mel_fmax \
        --clip $clip --log_base $log_base \
        --pitch_phoneme_averaging $pitch_phoneme_averaging \
        --energy_phoneme_averaging $energy_phoneme_averaging \
        --SSL_name $SSL_name --SSL_weight $SSL_weight --SSL_sample_rate $SSL_sample_rate \
        --mel_mode $mel_mode --pau_split $pau_split --n_jobs $n_jobs

    for s in ${datasets[@]}; do
        xrun python preprocess.py data/$s.list $wav_root $lab_root \
            $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate --filter_length $filter_length \
            --hop_length $hop_length --win_length $win_length \
            --n_mel_channels $n_mel_channels --mel_fmin $mel_fmin --mel_fmax $mel_fmax \
            --clip $clip --log_base $log_base \
            --pitch_phoneme_averaging $pitch_phoneme_averaging \
            --energy_phoneme_averaging $energy_phoneme_averaging  \
            --accent_info $accent_info
    done
    # preprocess実行時にのみcopyするようにする.
    mkdir -p $expdir/data
    cp -r data/*.list $expdir/data/
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature normalization"
    for typ in "fastspeech2"; do
       for inout in "out"; do
            for feat in "mel" "pitch" "energy"; do
                xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
                    $dump_org_dir/$train_set/${inout}_${typ}/${feat} \
                    $dump_org_dir/${inout}_${typ}_${feat}_scaler.joblib \
                    --speakers_list $speakers
            done
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "fastspeech2"; do
            for inout in "out" "in"; do
                if [ $inout == "in" ]; then
                    cp -r $dump_org_dir/$s/${inout}_${typ} $dump_norm_dir/$s/
                    continue
                fi
                for feat in "mel" "pitch" "energy" "duration"; do
                    if [ $feat == "duration" ]; then
                        cp -r $dump_org_dir/$s/${inout}_${typ}/${feat} $dump_norm_dir/$s/${inout}_${typ}
                        continue
                    fi
                    xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
                        $dump_org_dir/${inout}_${typ}_${feat}_scaler.joblib \
                        $dump_org_dir/$s/${inout}_${typ}/${feat} \
                        $dump_norm_dir/$s/${inout}_${typ}/${feat} --n_jobs $n_jobs
                done
            done
        done
    done

    if [ ! -z ${input_wav_paths} ] && [ -z ${SSL_name} ]; then
        echo "normalizing prosody embs..."
        xrun python $COMMON_ROOT/fit_scaler.py $dumpdir/${spk}_sr${sample_rate}/prosody_emb.list \
            $dumpdir/${spk}_sr${sample_rate}/prosody_emb \
            $dumpdir/${spk}_sr${sample_rate}/prosody_emb_scaler.joblib \
            --speakers_list $emb_speakers
        xrun python $COMMON_ROOT/preprocess_normalize.py $dumpdir/${spk}_sr${sample_rate}/prosody_emb.list \
            $dumpdir/${spk}_sr${sample_rate}/prosody_emb_scaler.joblib \
            $dumpdir/${spk}_sr${sample_rate}/prosody_emb \
            $dumpdir/${spk}_sr${sample_rate}/prosody_emb --n_jobs $n_jobs
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ -e $expdir/${vocoder_model} ] && [ ${overwrite} -le 0 ]; then
        echo "exp files are already exists. you cant run this exp."
        echo "if you want to overwrite, add --overwrite 1."
        exit 1
    fi
    # save config
    cp -r conf/train_hifigan $expdir/conf
    xrun python train_hifigan.py model=$vocoder_model tqdm=$tqdm \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$wav_root \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$wav_root \
        data.batch_size=$hifigan_data_batch_size \
        data.sampling_rate=$sample_rate \
        data.n_fft=$filter_length \
        data.num_mels=$n_mel_channels \
        data.hop_size=$hop_length \
        data.win_size=$win_length \
        data.fmin=$mel_fmin \
        data.fmax=$mel_fmax \
        train.out_dir=$expdir/${vocoder_model} \
        train.log_dir=tensorboard/${expname}_${vocoder_model} \
        train.nepochs=$hifigan_train_nepochs
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training fastspeech2"

    if [ -e $expdir/${acoustic_model} ] && [ ${overwrite} -le 0 ]; then
        echo "exp files are already exists. you cant run this exp."
        echo "if you want to overwrite, add --overwrite 1."
        exit 1
    fi

    # save config
    cp -r conf/train_fastspeech2 $expdir/conf
    if [ ! -z ${local_dir} ]; then
        echo "copy dataset to ${local_dir}"
        # copy data
        # first zip
        if [ ! -e ${dump_norm_dir}/${train_set}/in_fastspeech2.zip ]; then
            echo "zip ${dump_norm_dir}/${train_set}/in_fastspeech2.zip"
            zip -rq ${dump_norm_dir}/${train_set}/in_fastspeech2.zip $dump_norm_dir/$train_set/in_fastspeech2/
        fi
        if [ ! -e ${dump_norm_dir}/${train_set}/out_fastspeech2.zip ]; then
            echo "zip ${dump_norm_dir}/${train_set}/out_fastspeech2.zip"
            zip -rq ${dump_norm_dir}/${train_set}/out_fastspeech2.zip $dump_norm_dir/$train_set/out_fastspeech2/
        fi
        if [ ! -e ${dump_norm_dir}/${dev_set}/in_fastspeech2.zip ]; then
            echo "zip ${dump_norm_dir}/${dev_set}/in_fastspeech2.zip"
            zip -rq ${dump_norm_dir}/${dev_set}/in_fastspeech2.zip $dump_norm_dir/$dev_set/in_fastspeech2/
        fi
        if [ ! -e ${dump_norm_dir}/${dev_set}/out_fastspeech2.zip ]; then
            echo "zip ${dump_norm_dir}/${dev_set}/out_fastspeech2.zip"
            zip -rq ${dump_norm_dir}/${dev_set}/out_fastspeech2.zip $dump_norm_dir/$dev_set/out_fastspeech2/
        fi
        if [ ! -e ${dumpdir}/${spk}_sr${sample_rate}/text_emb.zip ]; then
            echo "zip ${dumpdir}/${spk}_sr${sample_rate}/text_emb.zip"
            zip -rq ${dumpdir}/${spk}_sr${sample_rate}/text_emb.zip $dumpdir/${spk}_sr${sample_rate}/text_emb/
        fi
        if [ -e ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb ] && [ ! -e ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb.zip ]; then
            echo "zip ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb.zip"
            zip -rq ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb.zip ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb/
        fi
        if [ -e ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb ] && [ ! -e ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb.zip ]; then
            echo "zip ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb.zip"
            zip -rq ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb.zip ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb/
        fi
        # unzip
        unzip -q  -d ${local_dir} ${dump_norm_dir}/${train_set}/in_fastspeech2.zip
        unzip -q  -d ${local_dir} ${dump_norm_dir}/${train_set}/out_fastspeech2.zip
        unzip -q -d ${local_dir} ${dump_norm_dir}/${dev_set}/in_fastspeech2.zip
        unzip -q -d ${local_dir} ${dump_norm_dir}/${dev_set}/out_fastspeech2.zip
        unzip -q -d ${local_dir} ${dumpdir}/${spk}_sr${sample_rate}/text_emb.zip
        if [ -e ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb.zip ]; then
            unzip -q -d ${local_dir} ${dumpdir}/${spk}_sr${sample_rate}/prosody_emb.zip
        fi
        if [ -e ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb.zip ]; then
            unzip -q -d ${local_dir} ${dumpdir}/${spk}_sr${sample_rate}/g_prosody_emb.zip
        fi
    fi
    xrun python train_fastspeech2wContexts.py model=$acoustic_model tqdm=$tqdm \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=${local_dir}$dump_norm_dir/$train_set/in_fastspeech2/ \
        data.train.out_dir=${local_dir}$dump_norm_dir/$train_set/out_fastspeech2/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=${local_dir}$dump_norm_dir/$dev_set/in_fastspeech2/ \
        data.dev.out_dir=${local_dir}$dump_norm_dir/$dev_set/out_fastspeech2/ \
        data.batch_size=$fastspeech2_data_batch_size \
        data.accent_info=$accent_info \
        data.acoustic_model=$acoustic_model \
        data.emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/text_emb \
        data.prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/prosody_emb \
        data.g_prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/g_prosody_emb \
        data.dialogue_info=$dialogue_info \
        data.use_hist_num=$use_hist_num \
        data.use_prosody_hist_num=$use_prosody_hist_num \
        data.use_local_prosody_hist_idx=$use_local_prosody_hist_idx \
        data.use_situation_text=$use_situation_text \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.nepochs=$fastspeech2_train_nepochs \
        train.sampling_rate=$sample_rate \
        train.mel_scaler_path=$dump_norm_dir/out_fastspeech2_mel_scaler.joblib \
        train.vocoder_name=$vocoder_model \
        train.vocoder_config=$vocoder_config \
        train.vocoder_weight_path=$vocoder_weight_base_path/$vocoder_eval_checkpoint \
        train.criterion.pitch_feature_level=$pitch_phoneme_averaging \
        train.criterion.energy_feature_level=$energy_phoneme_averaging \
        model.netG.pitch_feature_level=$pitch_phoneme_averaging \
        model.netG.energy_feature_level=$energy_phoneme_averaging \
        model.netG.n_mel_channel=$n_mel_channels \
        model.netG.accent_info=$accent_info \
        model.netG.mel_embedding_mode=$mel_mode \
        model.netG.pau_split_mode=$pau_split
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis waveforms by hifigan"

    if [ -e $expdir/synthesis_${acoustic_model}_${vocoder_model} ] && [ ${overwrite} -le 0 ]; then
        echo "synthesis files are already exists. you cant run this synthesis."
        echo "if you want to overwrite, add --overwrite 1."
        exit 1
    fi

    # save config
    cp -r conf/synthesis $expdir/conf
    if [ ! -z ${local_dir} ]; then
        echo "copy dataset to ${local_dir}"
        # input data dirs
        for s in ${testsets[@]}; do
            if [ ! -e $dump_norm_dir/$s/in_fastspeech2.zip ]; then
                echo "zip $dump_norm_dir/$s/in_fastspeech2.zip"
                zip -rq $dump_norm_dir/$s/in_fastspeech2.zip $dump_norm_dir/$s/in_fastspeech2/
            fi
            if [ ! -e $dump_norm_dir/$s/out_fastspeech2/mel.zip ]; then
                echo "zip $dump_norm_dir/$s/out_fastspeech2/mel.zip"
                zip -rq $dump_norm_dir/$s/out_fastspeech2/mel.zip $dump_norm_dir/$s/out_fastspeech2/mel
            fi
            unzip -q  -d ${local_dir} $dump_norm_dir/$s/in_fastspeech2.zip
            unzip -q  -d ${local_dir} $dump_norm_dir/$s/out_fastspeech2/mel.zip
        done
        if [ ! -e $dumpdir/${spk}_sr${sample_rate}/text_emb.zip ]; then
            echo "zip $dumpdir/${spk}_sr${sample_rate}/text_emb.zip"
            zip -rq $dumpdir/${spk}_sr${sample_rate}/text_emb.zip $dumpdir/${spk}_sr${sample_rate}/text_emb/
        fi
        if [ -e $dumpdir/${spk}_sr${sample_rate}/prosody_emb ] && [ ! -e $dumpdir/${spk}_sr${sample_rate}/prosody_emb.zip ]; then
            echo "zip $dumpdir/${spk}_sr${sample_rate}/prosody_emb.zip"
            zip -rq $dumpdir/${spk}_sr${sample_rate}/prosody_emb.zip $dumpdir/${spk}_sr${sample_rate}/prosody_emb/
        fi
        if [ -e $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb ] && [ ! -e $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb.zip ]; then
            echo "zip $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb.zip"
            zip -rq $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb.zip $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb/
        fi
        unzip -q -d ${local_dir} $dumpdir/${spk}_sr${sample_rate}/text_emb.zip
        if [ -e $dumpdir/${spk}_sr${sample_rate}/prosody_emb.zip ]; then
            unzip -q -d ${local_dir} $dumpdir/${spk}_sr${sample_rate}/prosody_emb.zip
            unzip -q -d ${local_dir} $dumpdir/${spk}_sr${sample_rate}/g_prosody_emb.zip
        fi
    fi
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=${local_dir}$dump_norm_dir/$s/in_fastspeech2 \
            in_mel_dir=${local_dir}$dump_norm_dir/$s/out_fastspeech2/mel \
            out_dir=${local_dir}$expdir/synthesis_${acoustic_model}_${vocoder_model}/$s \
            dialogue_info=$dialogue_info \
            emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/text_emb \
            use_hist_num=$use_hist_num \
            use_prosody_hist_num=$use_prosody_hist_num \
            prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/prosody_emb \
            g_prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/g_prosody_emb \
            use_local_prosody_hist_idx=$use_local_prosody_hist_idx \
            SSL_name=$SSL_name \
            SSL_weight=$SSL_weight \
            SSL_sample_rate=$SSL_sample_rate \
            mel_mode=$mel_mode \
            use_situation_text=$use_situation_text \
            sample_rate=$sample_rate \
            acoustic.checkpoint=$expdir/${acoustic_model}/$acoustic_eval_checkpoint \
            acoustic.out_scaler_path=$dump_norm_dir/out_fastspeech2_mel_scaler.joblib \
            acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
            vocoder.checkpoint=$vocoder_weight_base_path/$vocoder_eval_checkpoint \
            vocoder.model_yaml=$vocoder_config \
            reverse=$reverse num_eval_utts=$num_eval_utts
    done

    if [ ! -z ${local_dir} ]; then
        echo "copy results"
        for s in ${testsets[@]}; do
            mkdir -p $expdir/synthesis_${acoustic_model}_${vocoder_model}/$s/

            rsync -ah --no-i-r --info=progress2 ${local_dir}$expdir/synthesis_${acoustic_model}_${vocoder_model}/$s/ $expdir/synthesis_${acoustic_model}_${vocoder_model}/$s/
        done
    fi
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Calc dev loss by hist len"

    xrun python get_dev_loss_by_history.py model=$acoustic_model tqdm=$tqdm \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=${local_dir}$dump_norm_dir/$train_set/in_fastspeech2/ \
        data.train.out_dir=${local_dir}$dump_norm_dir/$train_set/out_fastspeech2/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=${local_dir}$dump_norm_dir/$dev_set/in_fastspeech2/ \
        data.dev.out_dir=${local_dir}$dump_norm_dir/$dev_set/out_fastspeech2/ \
        data.batch_size=$fastspeech2_data_batch_size \
        data.accent_info=$accent_info \
        data.acoustic_model=$acoustic_model \
        data.emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/text_emb \
        data.prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/prosody_emb \
        data.g_prosody_emb_dir=${local_dir}$dumpdir/${spk}_sr${sample_rate}/g_prosody_emb \
        data.dialogue_info=$dialogue_info \
        data.use_hist_num=$use_hist_num \
        data.use_local_prosody_hist_idx=$use_local_prosody_hist_idx \
        data.use_situation_text=$use_situation_text \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.nepochs=$fastspeech2_train_nepochs \
        train.sampling_rate=$sample_rate \
        train.mel_scaler_path=$dump_norm_dir/out_fastspeech2_mel_scaler.joblib \
        train.vocoder_name=$vocoder_model \
        train.vocoder_config=$vocoder_config \
        train.vocoder_weight_path=$vocoder_weight_base_path/$vocoder_eval_checkpoint \
        train.criterion.pitch_feature_level=$pitch_phoneme_averaging \
        train.criterion.energy_feature_level=$energy_phoneme_averaging \
        model.netG.pitch_feature_level=$pitch_phoneme_averaging \
        model.netG.energy_feature_level=$energy_phoneme_averaging \
        model.netG.n_mel_channel=$n_mel_channels \
        model.netG.accent_info=$accent_info \
        model.netG.mel_embedding_mode=$mel_mode \
        model.netG.pau_split_mode=$pau_split
fi

if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ]; then
    echo "Tuning fastspeech2 by optuna"
    mkdir -p $expdir/${acoustic_model}

    xrun python tuning_fastspeech2.py model=$acoustic_model tqdm=$tqdm \
        tuning=$tuning_config tuning.study_name=$expname\
        tuning.storage=sqlite:///../../../${expdir}/${acoustic_model}/optuna_study.db \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_fastspeech2/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_fastspeech2/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_fastspeech2/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_fastspeech2/ \
        data.batch_size=$fastspeech2_data_batch_size \
        data.accent_info=$accent_info \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.nepochs=$fastspeech2_train_nepochs \
        train.sampling_rate=$sample_rate \
        train.mel_scaler_path=$dump_norm_dir/out_fastspeech2_mel_scaler.joblib \
        train.vocoder_name=$vocoder_model \
        train.vocoder_config=$vocoder_config \
        train.vocoder_weight_path=$vocoder_weight_base_path/$vocoder_eval_checkpoint \
        train.criterion.pitch_feature_level=$pitch_phoneme_averaging \
        train.criterion.energy_feature_level=$energy_phoneme_averaging \
        model.netG.pitch_feature_level=$pitch_phoneme_averaging \
        model.netG.energy_feature_level=$energy_phoneme_averaging \
        model.netG.n_mel_channel=$n_mel_channels \
        model.netG.accent_info=$accent_info
    
    # save config
    cp -r conf/train_fastspeech2 $expdir/conf
fi

if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ]; then
    echo "Create tar.gz to share experiments"
    rm -rf tmp/exp
    mkdir -p tmp/exp/$expname
    rsync -avr $expdir/$acoustic_model tmp/exp/$expname/ --exclude "epoch*.pth"
    rsync -avr $vocoder_weight_base_path tmp/exp/$expname/ --exclude "epoch*.pth"
    rsync -avr $expdir/synthesis_${acoustic_model}_${vocoder_model} tmp/exp/$expname/ --exclude "epoch*.pth"
    cd tmp
    tar czvf fastspeech2_exp.tar.gz exp/
    mv fastspeech2_exp.tar.gz ..
    cd -
    rm -rf tmp
    echo "Please check fastspeech2_exp.tar.gz"
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${acoustic_model}_${vocoder_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
acoustic_model: ${acoustic_model}
vocoder_model: ${vocoder_model}
EOL

    # Stats
    python $COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/out_fastspeech2_mel_scaler.joblib $dst_dir

    # Acoustic model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
        $dst_dir/acoustic_model.pth
    cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

    # vocoder
    python $COMMON_ROOT/clean_checkpoint_state.py $vocoder_weight_base_path/$vocoder_eval_checkpoint \
        $dst_dir/vocoder_model.pth
    cp $vocoder_weight_base_path/model.yaml $dst_dir/vocoder_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
fi