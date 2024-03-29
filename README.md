
# 概要
## Overview

- https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data?select=metadata

不意にポットホールなどの道路障害物にぶつかったことはありませんか？また、ナビゲーションアプリで、より正確な位置情報や車線レベルの精度が得られたらと思ったことはありませんか？これらの機能やその他の新しい機能は、スマートフォンの測位サービスによって実現されています。機械学習と高精度GNSSアルゴリズムにより、この精度が向上し、何十億人ものAndroid携帯電話ユーザーに、よりきめ細かな測位体験を提供できるようになると期待されています。全地球測位衛星システム（GNSS）は、生の信号を提供し、GPSチップセットはそれを使って位置を計算します。現在の携帯電話では、3〜5メートルの位置精度しか得られません。これは多くの場合、便利ではありますが、"ビビり "の原因となります。多くのユースケースでは、その結果は、信頼できるほど細かくも安定してもいません。Android GPSチームが主催するこのコンテストは、ION GNSS+ 2021 Conferenceで発表されます。彼らは、スマートフォンのGNSS測位精度の研究を進め、人々が身の回りの世界をよりよくナビゲートできるようにすることを目指しています。このコンペティションでは、ホストチームが所有するAndroid携帯電話から収集したデータを使用して、可能であれば10cm、さらにはcm単位の分解能で位置情報を計算します。また、正確なグラウンドトゥルース、生のGPS測定値、近隣のGPSステーションからのアシスタンスデータを利用して、応募作品のトレーニングとテストを行います。成功すれば、より正確な位置情報を得ることができ、より細かい人間の行動の地理空間情報と、より細かい粒度のモバイルインターネットとの橋渡しをすることができます。モバイルユーザーは、より良い車線レベルの座標を得て、ロケーションベースのゲームの経験を強化し、交通安全上の問題の位置をより具体的に把握することができます。さらには、目的地への移動がより簡単になったことに気づくかもしれません。

謝辞 
Android GPSチームは、本チャレンジのデータセットにアシスタンスデータを提供してくださったVerizon Hyper Precise Location ServiceとSwift Navigation Skylark Correction Serviceに感謝します。

## Data

このチャレンジでは、GPS衛星からの信号、加速度計の測定値、ジャイロスコープの測定値など、携帯電話の位置を決定するのに役立つさまざまな機器からのデータを提供します。この課題では、車線レベルのマッピングなどの後処理に重点を置いて設計されているため、将来的にはルート上のデータを利用して、可能な限り正確な位置を生成することができます。また、複数の機種で構成される路線も多いため、隣接する機種の情報を利用して推定を行うこともできます。一般的なGNSS測位アルゴリズムの開発を促進するため、携帯電話内のGPSチップセットの位置情報は提供されません。これは、携帯電話のモデルなどによって異なるメーカー独自のアルゴリズムに基づいているためです。データ収集プロセスの詳細については、本稿をご参照ください。このデータセット／課題に基づいて作品を発表する場合は、競技規則に従って適切に引用してください。

# 古濵メモ

## カラム関連

- hDop,vDop：水平方向と垂直方向の測定誤差のこと

## ドメイン知識関連

- GNSSコンステレーションタイプ：GNSSの種類（参考URL：GNSSとは参照）
    - 数値変換には、metadata/constellation_type_mapping.csvを用いている

## 参考

### GNSSとは
- https://www.keisokuten.jp/static/sp_gnss.html

### GPSとは
- https://www.furuno.com/jp/gnss/technical/tec_what_gps#:~:text=DOP,%E7%B2%BE%E5%BA%A6%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%82%8A%E3%81%BE%E3%81%99%E3%80%82

# フォルダ構成

```
├── data                                    <- データ関連
│   ├── interim                             <- 作成途中のデータ
│   ├── processed                           <- 学習に使うデータ
│   ├── raw                                 <- 生データ
│   │   ├── metadata
│   │   ├── test
│   │   ├── train
│   │   ├── baseline_locations_test.csv     <- シンプルな方法で生成された推定座標（テスト）
│   │   ├── baseline_locations_train.csv    <- シンプルな方法で生成された推定座標（学習）
│   ├── submission                          <- 提出用データ
│   │   ├── sample_submission.csv
├── scripts                                 <- プログラム類
│   ├── furu                                <- 古プログラム
│   ├── gacky                               <- がっきープログラム
│   ├── sample                              <- サンプル
│   │   ├── generate.py                     <- データ作成（生データを分析しやすい整然データに→interimに保存）
│   │   ├── analyze.py                      <- 分析用スクリプト
│   │   ├── run.py                          <- 学習用スクリプト
│   │   ├── models                          <- モデル関連クラス
│   │   │   ├── model.py                    <- モデル基底クラス
│   │   │   ├── model_lgb.py                <- LightGBMクラス
│   │   │   ├── util.py                     <- 汎用的処理クラス
│   │   ├── config                          <- 汎用的処理クラス
│   │   │   ├── features                    <- 特徴量のカラム群
│   │   │   ├── params                      <- ハイパーパラメータ（コードに直接書いた方が良いかも）
├── models                                  <- 作成したモデル保存
├── config                                  <- 設定ファイル
│   ├── features                            <- 特徴ベクトル類

```

# 環境構築

## 前提

- 分析用の環境にはanaconda3使う
- pythonのバージョンは3.7.9を使う

## windows

1. [1]からAnaconda3-2020.11-Windows-x86=64.exeをダウンロードする
    - **ぶっちゃけ、anaconda入ってればなんでもいい**

2. インストーラーをNext押し続け（pathを通さないことを勧める）
3. タスクバーの虫眼鏡からAnaconda Promptと検索し、 起動する（以下、ターミナル=Anaconda Prompt）

    (なお、base環境はanaconda3の基底環境なので基本この環境で実行しない)

    ```
    (base) C:\Users\furuhama\OneDrive\document\Research\ADforDrone>
    ```

4. [2]などを参考に、ターミナルで`conda env create -n py37 -f py37env.yml`として、分析用の仮想環境を構築する
    - py37の部分は仮想環境の名前なので、お好みで
    - py37env.ymlはREADME.mdと同じディレクトリ階層にある

5. 作成した仮想環境に`conda activate py37`で入る
    ```
    (py37) C:\Users\furuhama\OneDrive\document\Research\ADforDrone>
    ```
6. 以上で準備完了

## 参考

- [1] Anaconda installer archive
  - https://repo.anaconda.com/archive/

- [2]【初心者向け】Anacondaで仮想環境を作ってみる
  - https://qiita.com/ozaki_physics/items/985188feb92570e5b82d