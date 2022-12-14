# フォルダ構成説明

- vc_tts_template
    - dnntts: ttslearnのdnntts.
    - fastspeech2: original. 2021/09追加.
    - frontend: テキストを処理するためのフォルダ.
    - pretrained:
        各フォルダにある`tts.py`用の, pretrainモデルのダウンロード取得担当
    - tacotron: ttslearnのtacotron2
    - wavenet: ttslearnのwavenet
    - dsp.py:
        degital signal processorの略. 各種信号処理関数が入っている.
    - logger.py:
        その名の通りlogger実装用
    - train_utils.py:
        train周りの便利関数集
    - utils.py:
        便利関数

## todo
- eval_model実装
- hifigan実装

## モデル実装ルール
- そのフォルダ内だけで完結するように, 引数はここに関するものをすべて生で受け取る.
    - 例: configという引数などは作らない. すべてdim=など書く.

- gen.py(synthesis用)
    - 例: tacotronのものを見るのが一番きれい.
    - 簡単に言えば, 入力データとモデルを受け取って, 出力データを返す関数を用意する. 出力フォルダへの保存などはすべて別の場所で. とにかくモデル絡みの計算はここで完結させる.
- tts.py(気軽に実行できる用)
    - 逆にこちらは, model_dirだけ受け取ったらそれ以降のすべての処理を行い最終完成品(音声)を出力する. 一番最後の実装になると思われる.
- collate_fn.py
    - 各dataloaderに渡すためのcollate_fn. これを書けばdataloader周りはdone.

- モデルファイル
    - (model名).py: 最終形
    - (ブロック名).py: 最終系で用いる, 大きなモジュール単位
        - 例: encoder, decoder, variance_adapter
    - layers.py: 各ブロックで利用する一番大きな単位
    - sublayers.py: layersで利用するlayer
    - modules.py: sublayersで利用するlayer

## 汎用ファイル詳細説明
- frontend
    - openjtalk.py: openjtalkで用いられる語彙を基準としたtext処理関数. symbolの個数などはここから得られる.
    - text.py: ↑これの英語version.
- dsp.py
    - f0_to_lf0: f0をlog付けて返す. 0の部分は0のままにする.
    - lf0_to_f0: logf0をf0にする. vuvで0のところを復元する.
    - compute_delta: 動的特徴量を計算する. 特徴量の次元ごとに. 窓を与える必要あり.
    - world_log_f0_vuv: worldを用いて, 音声からlogF0とvuvを抽出する. 但し, logF0は動的なそれも取り出してきちゃうので注意. @wavenet
    - world_spss_params: mgc, logf0, bap, vuvを計算し, まとめて返す関数. @dnntts
    - mulaw_quantize: Mu-Lawによって音声を圧縮&離散化して返す関数.
    - inv_mulaw_quantize: mulaw_quantizeの逆
    - logspectrogram: log_spectrogram, つまり, stft→log10をとる. clipなどはtacotronの論文ベース(tacotron2ではない).
    - logmelspectrogram: stft→mel→log10. @tacotron2
    - logmelspectrogram_to_audio: griffinlimアルゴリズムを用いて, logmelをaudioにする.

- logger.py
    - getLogger: fileの場所とかを指定してloggerを返してくれる関数.

- train_utils.py
    - free_tensors_memory: 不要になった変数を解放します.
    - check_grad_flow: gradがinfになったやつを記録しておいて, log吐くやつ
    - get_epochs_with_optional_tqdm: epoch数を指定して, tqdmか否かを選べば, 1から始まるepochのイテレータを返す.
    - num_trainable_params: モデルを渡したら, パラメタ数を返してくれる
    - set_epochs_based_on_max_steps_: max_step指定した訓練回数をepochのそれに変換する.
    - save_checkpoint: 保存用関数. is_baseを使うことで, bestと名前を付けてくれる.
    - ensure_divisible_by: 与えられた特徴量の長さがNで割り切れない時, 削って返す. 正直, paddingしてほしいので使えない. 実際使われていない.
    - moving_average_: パラメータの移動平均を計算し, テスト用モデルに再代入してあげる. @wavenet
    - plot_grad_flow: gradientを可視化してくれる. 最高.
    - plot_attention: attention weightを入れたらfigを返してくれる.
    - plot_mels: melに特化したわけではないが, 2次元の特徴量を複数渡したときに縦にくっつけて表示してくれる.
    - plot_2d_feats: 二次元の特徴量を図示してくれる.
    - get_vocoder: vocoderのモデルインスタンスを返す.
    - vocoder_infer: melを渡したらwavにしてくれる. get_vocoderと組み合わせて利用.
    - setup: configとcollate_fnを用意するだけですべてインスタンス化して返してくれる. その代わり, ちゃんとconfigを書く必要がある.

- utils.py
    - adaptive_load_state_dict: 読み込めるkeyだけ読み込むload_state_dict
    - optional_tqdm: tqdm, tqdm-notebookどちらかを返してくれる関数. 基本使わないかも.
    - make_pad_mask: pad部分がTrueになるようなmaskを返す.
    - make_non_pad_mask: ↑これの逆.
    - load_utt_list: txtファイルを読み, その中のfile名リストを返す.
    - init_seed: numpy, tensorなど諸々のseedセット.
    - pad_1d: 1次元tensorのpad
    - pad_2d: 2次元tensorのpad
    - pad: batch対応の, ネットワーク内で利用する想定のpadding.
    - StandardScaler: 自前のmean, stdで正規化してくれるやつ. いつ使う?