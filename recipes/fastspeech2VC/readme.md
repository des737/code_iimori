# fastspeech2VC利用方法.

- todo
    - sent_durationはdata_preparationまで終了
    - 次はRichProsodyと同じwGMMを実装

## 事前準備
1. 利用したいコーパスのwavをどこかに置く.
    - srは変更する必要なし.
    - ファイル名は, src_tgt_wavnameにすること.
        - 例: jsut_jsss_BASIC5000_0001.wav
    - `./run.sh`を実行する場所からの相対パスを指定する.
    - 例: "../../../dataset/out_JSUT/JSUT/wav"
3. 終わり. ./run.shを実行していってください.

**注意**
- multi-speakerにしたい場合は, netGのspeakersに事前に用意したspeakerのdictionalyを追加すること.

- emotion=Trueにしたい場合は, netGのemotionsに事前に用意したemotionのdictionalyを追加すること. 
    - そして, ファイル名をファイル名_src_tgtにする.
    - 例: jsut_jsss_BASIC5000_0001_joy_sorrow.wav

- train/dev/eval splitは毎回shuffleしている. 実験を完全再現したいのであれば, stage 0は実行せず, gitに入っているものをそのまま利用すること.

- 実験名管理について
    - spk: 全体の名前. preprocessed_data名にもなるため, データそのものの変更をするならここを変える.
    - tag: 毎度おなじみexp_name. 実験名の変更です. 訓練時.


## memo
- sent_durationは, 推論時はあってもなくてもよい.
    - 入力しない場合は, GSTとしてふるまうように設計する.
    - 実装自体は, wGMMと同様のやり方.