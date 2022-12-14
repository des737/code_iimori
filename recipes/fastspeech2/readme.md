# fastspeech2利用方法.
## 事前準備
1. 利用したいコーパスのwavをどこかに置く.
    - srは変更する必要なし.
    - `./run.sh`を実行する場所からの相対パスを指定する.
    - 例: "../../../dataset/out_JSUT/JSUT/wav"
2. 利用したいコーパスのtextgridをどこかに置く.
    - 例: "../../../dataset/out_JSUT/textgrid"
    - textgridのファイルを基準にファイルリストを作成します.
3. 終わり. ./run.shを実行していってください.

**注意**
- multi-speakerにしたい場合は, ファイル名の前に話者名を「_」を付けて追加すること.
    - 例: `JSUT_BASIC5000_0001.wav`
    - そして, netGのspeakersに事前に用意したspeakerのdictionalyを追加すること.
- emotion embを付けたい場合は, ファイル名の**後**に感情を「_」をつけて追加.
    - 例: `JSUT_BASIC5000_0001_悲しみ.wav`
    - そして, fastspeech2wEmotionのnetGのemotionsに事前に用意したspeakerのdictionalyを追加すること.

- statsに関しては, dump以下に全データ通してのそれをjsonとして吐きだすので, それを参考に入力. 

- train/dev/eval splitは毎回shuffleしている. 実験を完全再現したいのであれば, stage 0は実行せず, gitに入っているものをそのまま利用すること.

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

## 2021-12-24: pitch, energy量子化の問題点
- pretrainとfinetuningでbinは固定する必要がある
    - そうしないと, pretrainで得た特徴がfinetuningに役に立たない
    - 例: pretrain(LINE), finetuning(JSUT)
        - pretrainにてbin=255はほとんど外れ値. かなり演技が入っている.
        - これをJSUTのbin幅に置き換えてしまうと, 幅が狭い分簡単にbin=255に行くが, それはpretrainで言うところのかなりの演技.
        - つまり, pitch幅が違うと意味がないので.
- さらに, 標準化している時点で↑これは成立しない
    - なぜなら, JSUTなど, stdが小さいほうが過大評価されるため.
    - 実際, pitchの標準化後のQ3値は,
        - JSUT+NICT+LINE: {"pitch_min": -0.8101340727883142, "pitch_max": 0.700753939829083, "energy_min": -0.7406348586082458, "energy_max": 0.4284781515598297}
        - JSUT: {"pitch_min": -0.8220028797645906, "pitch_max": 0.7265703738211664, "energy_min": -0.8273443728685379, "energy_max": 0.6603641211986542}
    - とかなので, もう固定しようがあてにならない.
- つまり, binを使うこと自体が微妙の可能性あり
    - 実際, espnetでは使っていない
    - https://espnet.github.io/espnet/_modules/espnet2/tts/fastspeech2/fastspeech2.html
    - 議論されている: https://github.com/espnet/espnet/issues/2019
- なしバージョンも作成せよ!!!
    - 作成. ststs=Noneで動くようになります

## 2021-12-24: multi-speakerについて
- 「話者ごとに正規化しろ!!!」
- 当然のことを忘れていた.
- pretrainにもmulti-speakerは使うし, PEPCEにも使うので, これの実装は必須.
- 対応しました. scalerとして新クラスを定義.
- configでspeakersを設定しましょう.

- pretrainでMS対応がいいかは、状況に依りそう.
    - finetuning先がsingle-speaker: 正規化した方がよさそう
    - finetuning先もMS対応 or Single-shotを想定: 全体で正規化した方がよさそう.
    - もちろん, やってみないとわからないと思うからとりあえずやるべき.