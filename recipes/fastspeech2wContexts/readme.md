# fastspeech2wContexts利用方法.
## 事前準備
0. wav, textgridのファイル名の先頭に、「話者名_」というprefixを付けてください。
1. 利用したいコーパスのwavをどこかに置く.
    - srは変更する必要なし.
    - `./run.sh`を実行する場所からの相対パスを指定する.
    - 例: "../../../dataset/out_JSUT/JSUT/wav"
2. 利用したいコーパスのtextgridをどこかに置く.
    - 例: "../../../dataset/out_JSUT/textgrid"
    - textgridのファイルを基準にファイルリストを作成します.
3. 実際に発話生成を行いたい話者の音声ファイル名の書かれたlistを作成し、`data`以下に配置してください.
    - 注意点
    - train/dev/evalの3つに分割してください。
    - 同一対話が同じ分割内に入るようにしてください。それを守らないと、リークが発生します。
4. 対話情報を記録したtextファイルを用意します。以下の形式で書いてください。
適当な場所に配置した後、1,2同様`./run.sh`からの相対パスを指定してください。

Example
```
Teacher_LD01-Dialogue-01-Teacher-Turn-02_joy:1:2
Student_LD01-Dialogue-01-Teacher-Turn-03_joy:1:3
Teacher_LD01-Dialogue-01-Teacher-Turn-04_joy:1:4
```

それぞれ、  
(ファイル名):(対話ID):(対話内ID)  
になります。

5. speakers_dictを作成してください. 但し、必ず話者「"pad"」を用意し、IDは-1としてください。


## その他情報
- emotion embを付けたい場合は, ファイル名の**後**に感情を「_」をつけて追加.
    - 例: `JSUT_BASIC5000_0001_悲しみ.wav`
    - そして, fastspeech2wEmotionのnetGのemotionsに事前に用意したspeakerのdictionalyを追加すること.
    - こちらも, speakers同様に, {"pad": 0}を用意してください。

- statsに関しては, dump以下に全データ通してのそれをjsonとして吐きだすので, それを参考に入力しましょう。

- lossは, `fastspeech2wGMMwContexts`を利用する場合, `fastspeech2wGMM`のlossをそのまま利用します.

- 実験名管理について
    - spk: 全体の名前. preprocessed_data名にもなるため, データそのものの変更をするならここを変える.
    - tag: 毎度おなじみexp_name. 実験名の変更です. 訓練時.

## 2021-09-29: fastspeech2wGMM追加
### 仕様
- devのlossは, 正解melは与えて, pitch, energyを与えないで計算されているが, 正解melはloss計算用の正解prosody embを計算するためだけに用いられて, 出力のmel自体はduration以外inference時と同じ設定で計算されている.

### 使い方
- configにおいて, 以下のモデル名を指定
    - fastspeech2wGMM
- train_fastspeech2 configにおいて, 以下のlossを指定
    - vc_tts_template.fastspeech2wGMM.loss.FastSpeech2Loss
    - betaを決定
- 後は同様

### ttsの利用方法
- fastspeech2のtts.pyを利用してください.
