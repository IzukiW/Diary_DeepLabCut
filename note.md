
1. [25/06/12](#250612)
2. [25/06/13](#250613)
3. [25/06/15](#250615)
4. [25/06/16](#250616)
5. [25/06/20](#250620)
6. [25/06/23](#250623)
      1. [Intersection over Union(IoU)](#intersection-over-unioniou)
      2. [Average precision(AP)とAverage Recall(AR)](#average-precisionapとaverage-recallar)
7. [25/06/24](#250624)
8. [25/06/25](#250625)
9. [25/06/26](#250626)
   1. [Task after meeting](#task-after-meeting)
10. [25/06/27](#250627)
11. [25/06/28](#250628)
12. [25/06/30](#250630)
13. [25/07/01](#250701)
14. [25/07/02](#250702)
15. [25/07/03](#250703)
16. [25/07/04](#250704)
17. [25/07/08](#250708)

---

## 25/06/12 

task
- 必要な教師データのフレーム数の把握
- 画像処理の下限

**Done**
1. anaconda上に仮想環境DEEPLABCUTを作成
1. deeplabcutに必要なcuda12.6(local), cuDNN(仮想環境)等をinstall
1. CUDAのpath通し

GPU version確認 >`nvidia-smi`
CUDA version確認 >`nvcc -V`



**Trouble**
1. `python -m deeplabcut` で起動しない -> 仮想環境(DEEPLABCUT)にpipがinstallされていなかった -> installしたら謎エラーを吐かれた
<details><summary>error表示1 FileNotFoundError"などとほざいている </summary>
```
>pip uninstall numpy
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\Scripts\pip-script.py", line 9, in <module>
    sys.exit(main())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main.py", line 64, in main
    cmd_name, cmd_args = parse_command(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main_parser.py", line 78, in parse_command
    general_options, args_else = parser.parse_args(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\optparse.py", line 1371, in parse_args
    values = self.get_default_values()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\parser.py", line 279, in get_default_values
    self.config.load()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 124, in load
    self._load_config_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 246, in _load_config_files
    config_files = dict(self.iter_config_files())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 339, in iter_config_files
    config_files = get_configuration_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 70, in get_configuration_files
    os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\utils\appdirs.py", line 48, in site_config_dirs
    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\__init__.py", line 146, in site_config_dir
    ).site_config_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 67, in site_config_dir
    return self.site_data_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 56, in site_data_dir
    path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 209, in get_win_folder_from_registry
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
FileNotFoundError: [WinError 2] The system cannot find the file specified
```
</details>


`conda install pip`したのに治らない
<details><summary>error表示2 同じ表示だね </summary>
```
>pip --verison
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\Scripts\pip-script.py", line 9, in <module>
    sys.exit(main())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main.py", line 64, in main
    cmd_name, cmd_args = parse_command(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main_parser.py", line 78, in parse_command
    general_options, args_else = parser.parse_args(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\optparse.py", line 1371, in parse_args
    values = self.get_default_values()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\parser.py", line 279, in get_default_values
    self.config.load()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 124, in load
    self._load_config_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 246, in _load_config_files
    config_files = dict(self.iter_config_files())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 339, in iter_config_files
    config_files = get_configuration_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 70, in get_configuration_files
    os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\utils\appdirs.py", line 48, in site_config_dirs
    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\__init__.py", line 146, in site_config_dir
    ).site_config_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 67, in site_config_dir
    return self.site_data_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 56, in site_data_dir
    path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 209, in get_win_folder_from_registry
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
FileNotFoundError: [WinError 2] The system cannot find the file specified
```
</details>

(DEEPLABCUT)上のpipをuninstallしたらエラーが消えた。baseのpipとconflictしてたのだろうか？


## 25/06/13

昨日は通らなかった`nvcc -V`が通るようになった。
```
C:\Users\satie>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
```

**Done**
- pytorchのinstall
- deeplabcut GUI版+model zooのinstall
- GUIでlabeling(manubrium_baseだけできなかったけど...)

installしたPytorch
![PyTorch version](Screenshot_13-6-2025_1990_pytorch.org.jpeg)

install後公式文書通りの応答が帰ってきた
```
#in my env(DEEPLABCUT)

(DEEPLABCUT) C:\Users\satie>python -c "import torch; print(torch.cuda.is_available())"
True
```

DeepLabCutは`pip install deeplabcut[gui,modelzoo]`で仮想環境上に置いた



[公式ドキュメント](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html)にならって
```py
ipython
import deeplabcut
```
すると"ModuleNotFoundError"が次々出てくる
<details><summary>error表示3 ruamel.yamlがありません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\utils\auxiliaryfunctions.py:31
     29 import numpy as np
     30 import pandas as pd
---> 31 import ruamel.yaml.representer
     32 import yaml
     33 from ruamel.yaml import YAML

ModuleNotFoundError: No module named 'ruamel'
```
</details>

`conda install ruamel.yaml` で対処

<details><summary>error表示 filelockがありません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\huggingface_hub\utils\_fixes.py:22
     19 from typing import Callable, Generator, Optional, Union
     21 import yaml
---> 22 from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout
     24 from .. import constants
     25 from . import logging

ModuleNotFoundError: No module named 'filelock'
```
</details>

`conda install conda-forge::filelock` で対処

<details><summary>error表示5 sympyがinstallできません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\torch\utils\_sympy\functions.py:10
      7 from typing_extensions import TypeVarTuple, Unpack
      9 import sympy
---> 10 from sympy import S
     11 from sympy.core import sympify
     12 from sympy.core.expr import Expr

ImportError: cannot import name 'S' from 'sympy' (unknown location)
```
</details>

`conda install sympy` で対処

エラーが出なくなった！！
anacondaからDeepLabCut(GUI)の起動方法は2つ
```
ipython
import deeplabcut
deeplabcut.launch_dlc()
```
または
```
python -m deeplabcut
```
![DeepLabCutの起動画面](<Screenshot 2025-06-13 201301.png>)

GUIにしたがってprojectを作成

![alt text](<Screenshot 2025-06-13 202642.png>)


## 25/06/15

**Done**
- DSC_3189_#1~5, 3197_#1~2のlabeling 
(4points: manubrium base, mouth, prey, tentacle base)
- DSC_3197/3237_#1~5のframe抽出
- dlc3.0の[Tutorial](https://youtu.be/ofFx0vTMSxE?si=hsLphcDQQz36v3cA)を見る(途中)
- `Create training dataset`タブでラベル済みの140データからなるdatasetを作成
- `Train Network`を実行


前回の続きなので、トップ画面から`Load Project`を選択 > config.yamlをopen

1video当たり20frames抽出する設定
20framesに4point打つのにかかった時間: 10分弱

Trainの様子


![Training中](<Screenshot 2025-06-15 222717.png>)

## 25/06/16
**Done**
- `C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\dlc-models-pytorch\iteration-0\Cladonema_starvedJun13-trainset95shuffle1\train\train.txt`でtraining中の出力を見る
- [beginner-guide> analyze video](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/video-analysis.html)まで進んだ
- traind modelを3231_roi_#5に適用したが全然trackしていなかった
- 一部のvideoを400×400にcrop。imagej使用



<details><summary> training progress </summary>

```py
2025-06-15 22:25:37 Training with configuration:
# ~~中略~~
2025-06-15 22:25:37 train_settings:
2025-06-15 22:25:37   batch_size: 8
2025-06-15 22:25:37   dataloader_workers: 0
2025-06-15 22:25:37   dataloader_pin_memory: False
2025-06-15 22:25:37   display_iters: 1000
2025-06-15 22:25:37   epochs: 200
2025-06-15 22:25:37   seed: 42
# ~~中略~~
2025-06-15 22:25:42 Using 133 images and 7 for testing
2025-06-15 22:25:42 
Starting pose model training...
--------------------------------------------------
2025-06-15 22:25:54 Epoch 1/200 (lr=0.0005), train loss 0.01532
2025-06-15 22:26:03 Epoch 2/200 (lr=0.0005), train loss 0.01283
2025-06-15 22:26:12 Epoch 3/200 (lr=0.0005), train loss 0.01031
2025-06-15 22:26:20 Epoch 4/200 (lr=0.0005), train loss 0.00726
2025-06-15 22:26:28 Epoch 5/200 (lr=0.0005), train loss 0.00659
# ~~中略~~
2025-06-15 22:53:44 Epoch 196/200 (lr=1e-05), train loss 0.00051
2025-06-15 22:53:52 Epoch 197/200 (lr=1e-05), train loss 0.00053
2025-06-15 22:54:01 Epoch 198/200 (lr=1e-05), train loss 0.00051
2025-06-15 22:54:09 Epoch 199/200 (lr=1e-05), train loss 0.00055
2025-06-15 22:54:18 Training for epoch 200 done, starting evaluation
2025-06-15 22:54:19 Epoch 200/200 (lr=1e-05), train loss 0.00048, valid loss 0.00180
2025-06-15 22:54:19 Model performance:
2025-06-15 22:54:19   metrics/test.rmse:           3.37
2025-06-15 22:54:19   metrics/test.rmse_pcutoff:   2.30
2025-06-15 22:54:19   metrics/test.mAP:           82.89
2025-06-15 22:54:19   metrics/test.mAR:           87.14
```
</details>

analyze実行
<details><summary> analyzing progress (6min 56s) </summary>

```py
NFO:console:Starting to analyze C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi
INFO:console:Video metadata:
  Overall # of frames:    3597
  Duration of video [s]:  120.02
  fps:                    29.97
  resolution:             w=1500, h=1080

INFO:console:Running pose prediction with batch size 8
100%|███████████████████████████████████████████████████████████████████| 3597/3597 [06:56<00:00,  8.63it/s]
INFO:console:Saving results in C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5DLC_Resnet50_Cladonema_starvedJun13shuffle1_snapshot_110.h5 and C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5DLC_Resnet50_Cladonema_starvedJun13shuffle1_snapshot_110_full.pickle
INFO:console:The videos are analyzed. Now your research can truly start!
You can create labeled videos with 'create_labeled_video'.
If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames.

INFO:console:Filtering with median model C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi
INFO:console:Saving filtered csv poses!
```
</details>

`Create video`
<details><summary> creating progress (48s) </summary>

```py
INFO:console:Filtering with median model C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi
INFO:console:Saving filtered csv poses!
INFO:console:Loading
INFO:console:
INFO:console:C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi
INFO:console:
INFO:console:and data.
INFO:console:Plots created! Please check the directory "plot-poses" within the video directory
100%|███████████████████████████████████████████████████████████████████| 3597/3597 [00:48<00:00, 74.06it/s]
INFO:console:Starting to process video: C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi
INFO:console:Loading C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\videos\DSC_3231_roi_#5.avi and data.
INFO:console:Duration of video [s]: 120.02, recorded with 29.97 fps!
INFO:console:Overall # of frames: 3597 with cropped frame dimensions: 1500 1080
INFO:console:Generating frames and creating video.
INFO:console:Labeled videos created.
```
</details>

## 25/06/20
**Done**
- New project: "C:\Users\satie\Desktop\izuki_temp\Cladonema_starved-Vlad&Genta-2025-06-13\Cladonema_starved_crop-Vlad&Genta-2025-06-20" 作成
  <br> <- 400×400でcropした動画
- DSC_3190~DSC_3206 #1~5を400×400でcrop->"Enhance Contrast"をかけて保存
- DSC_3174_#1~3をlabel
- DSC_3175/3176_#1~5をExtract frames済み


## 25/06/23
**Done**
- DSC_3207~DSC_3219 #1~5を400×400でcrop->"Enhance Contrast"をかけて保存<br>
  ⇒これで53videosが用意できた
- 結果の解釈についてのお勉強(途中、別ファイルにしてtexみたいに後から挿入してもいいかも)



**お勉強** <br>
pytorchを用いた場合の"CombinedEvaluation-results.csv"の解釈について 

![CombinedEvaluation-results.csv](Screenshot_2025-06-23_142033.png)

<details><summary> `train mAP`や`train mAR`とは何ぞや？</summary>

そもそもとして
deeplabcutは

#### Intersection over Union(IoU)
IoUは実際に物体が存在する範囲(正解範囲)とモデルが予測した物体の範囲(予測範囲)の重なり具合を表す指標のこと。
$$
\text{IoU} =\frac{\text{Intersection}}{\text{Union}}
=\frac{\text{重なっている範囲}}{\text{正解範囲と予測範囲の和集合}}
$$
このIoUが閾値を超えていれば検出成功(T)、下回れば検出失敗となる(F)。
検出結果を基に以下の４つの判定を下す。

- TP(True Positive): 物体がある場所を正しく予測できた。
- FP(False Positive): 物体がない場所を誤って予測した。偽陽性。
- FN(False Negative): 物体があった場所に予測をしなかった。偽陰性。
- TN(True Negative): 物体のない場所に予測をつけなかった。

表にするとこんな感じ↓

|  | 物体がある場所を | 物体がない場所を |
| ------- | :-------: | :-------: |
| **予測した** | TP | FP |
| **除外した** | FN | TN |

この判定を基に、PrecisionとRecallを計算。
```math
\begin{align}
\text{Precision} &=\frac{TP}{TP+FP}=\frac{予測が当たった}{物体があると予測した} \\[10pt]
\text{Recall} &= \frac{TP}{TP+FN}=\frac{予測が当たった}{実際に物体が存在した} \\
\end{align}
```

#### Average precision(AP)とAverage Recall(AR) 
テストデータで実験する。IoU閾値は0.5。
信頼度スコアは物体検知の結果がどの程度正確かというのを物体検知モデル自体が出力した値。
信頼度が高い順にindexを振った。

|No.|信頼度|IoU|正解|TP|FP|Precision|Recall|
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|1|0.95|0.88|T|1|0|1.0000|0.1429|
|2|0.88|0.76|T|2|0|1.0000|0.2857|
|3|0.84|0.69|T|3|0|1.0000|0.4286|
|4|0.77|0.81|T|4|0|1.0000|0.5714|
|5|0.72|0.54|T|5|0|1.0000|0.7143|
|6|0.65|0.48|F|5|1|0.8333|0.7143|
|7|0.61|0.62|T|6|1|0.8571|0.8571|
|8|0.53|0.51|T|7|1|0.8750|1.0000|
|9|0.44|0.39|F|7|2|0.7778|1.0000|
|10|0.38|0.46|F|7|3|0.7000|1.0000|

PrecisionとRecallの計算は以下の通り。正解の数(=正しい物体の数)は既知とする。
1. No.1
    - $\text{Precision}= 1/1=1$
    - $\text{Recall}= 1/7=0.1429$

1. No.2
    - $\text{Precision}= 2/2=1$
    - $\text{Recall}= 2/7=0.2857$

1. No.3
    - $\text{Precision}= 3/3=1$
    - $\text{Recall}= 3/7=0.4286$    

この表でPrecision-Recallカーブを書く。<br>
<img src="https://storage.googleapis.com/zenn-user-upload/6778d7a145ae-20240623.png" width="60%">



dlcではCOCO methodを採用しているよう([deeplabcut/core/metrics/bbox.py](https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/core/metrics/bbox.py))


Reference
- [物体検出モデルの精度評価を理解して実際に算出する](https://qiita.com/unyacat/items/1245bf595e53e79b5d4a)
- [【物体検出の評価指標】mAP ( mean Average Precision ) の算出方法](https://qiita.com/cv_carnavi/items/08e11426e2fac8433fed)
- [mAP (mean Average Precision) for Object Detection](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Mean-Average-Precision (mAP)](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)

</details>

どうなったら良い値に収束したといえる？
[このissue](https://forum.image.sc/t/evaluation-results/105920)が参考になった。
あと[このスライド](https://speakerdeck.com/eqs/ji-jie-xue-xi-falseji-chu-karali-jie-suru-deeplabcutfalseyuan-li)。

よく用いられる指標がRMSE(平均二乗誤差)。$y_i$が元データで$\hat{y_i}$が予測値。
次元は元データと一緒で、ずれ具合を表すパラメータ。
```math
\begin{equation*}
RMSE =\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
\end{equation*}
```


## 25/06/24
**Done**
- DSC_3179~3219_#1~5をExtract frames済み
- DSC_3175~3204の_#1~5をlabel(6videos, 30trials, 600frames)

`Label frames`tabで`check labels`を実行するとlabel済みのフォルダに対し、label付き画像の入ったフォルダを生成してくれる。これでどこまでlabel打ったか迷子にならなそう。


## 25/06/25
**Done**
-  DSC_3205の_#1~2をlabel <br>
  -> 25/06/25 13:27(JST)時点で計700framesがlabel済み。これで一回train回して時間と精度を測る。
- `Train network`を2回回す
  - 1回目 635 for training, 35 for test. 200 epochs. 1h34m43s. test RMSE=1.34. train RMSE=5.91.
  - 2回目


Error集

<details><summary> shuffle2のevaluation結果が"CombinedEvaluation-results.csv"に保存されない</summary>

```py
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:11<00:00, 54.96it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 55.43it/s]
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\gui\tabs\evaluate_network.py", line 207, in evaluate_network
    deeplabcut.evaluate_network(
  # 中略
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\satie\\Desktop\\izuki_temp\\Cladonema_starved_crop-Vlad&Genta-2025-06-20\\evaluation-results-pytorch\\iteration-0\\CombinedEvaluation-results.csv'
```
⇒開きっぱなしにしていた"CombinedEvaluation-results.csv"のファイルを閉じてから実行するときちんと上書きされた。
</details>

<details><summary> trainingが進まない</summary>

```py
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\gui\utils.py", line 25, in run
    self.func()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\compat.py", line 900, in analyze_videos
    engine = get_shuffle_engine(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\generate_training_dataset\metadata.py", line 414, in get_shuffle_engine
    shuffle_metadata = metadata.get(trainingsetindex, shuffle)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\generate_training_dataset\metadata.py", line 206, in get
    raise ValueError(
ValueError: Could not find a shuffle with trainingset fraction 0.9 and index 1
WARNING: QThread: Destroyed while thread is still running
```
⇒config.yamlのtrain_fraction値を0.95に直して実行したら出来た。shuffle1はtrain_fractionが0.95だったのに、shuffle2でconfig.yamlの値を0.9に変更し戻さずに実行したからerrorが出た模様。

<img src="Screenshot_2025-06-25_195449.png" width="50%">
</details>

analyze結果
analyze~createでどっちも１分弱ぐらいかかった。

<details><summary> shuffle1</summary>

```py
INFO:console:Analyzing videos with C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\dlc-models-pytorch\iteration-0\Cladonema_starved_cropJun20-trainset95shuffle1\train\snapshot-best-190.pt
  0%|                                                                                         | 0/3597 [00:00<?, ?it/s]INFO:console:Starting to analyze C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Video metadata:
  Overall # of frames:    3597
  Duration of video [s]:  119.90
  fps:                    30.0
  resolution:             w=400, h=400

INFO:console:Running pose prediction with batch size 8
100%|██████████████████████████████████████████████████████████████████████████████| 3597/3597 [00:47<00:00, 75.95it/s]
INFO:console:Saving results in C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_cropDLC_Resnet50_Cladonema_starved_cropJun20shuffle1_snapshot_190.h5 and C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_cropDLC_Resnet50_Cladonema_starved_cropJun20shuffle1_snapshot_190_full.pickle
INFO:console:The videos are analyzed. Now your research can truly start!
You can create labeled videos with 'create_labeled_video'.
If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames.

INFO:console:Filtering with median model C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Saving filtered csv poses!
100%|█████████████████████████████████████████████████████████████████████████████| 3597/3597 [00:06<00:00, 560.31it/s]
INFO:console:Starting to process video: C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Loading C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi and data.
INFO:console:Duration of video [s]: 119.9, recorded with 30.0 fps!
INFO:console:Overall # of frames: 3597 with cropped frame dimensions: 400 400
INFO:console:Generating frames and creating video.
INFO:console:Labeled videos created.
INFO:console:Loading
INFO:console:
INFO:console:C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:
INFO:console:and data.
INFO:console:Plots created! Please check the directory "plot-poses" within the video directory
```
</details>


<details><summary> shuffle2</summary>

```py
INFO:console:Analyzing videos with C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\dlc-models-pytorch\iteration-0\Cladonema_starved_cropJun20-trainset90shuffle2\train\snapshot-best-100.pt
  0%|                                                                                         | 0/3597 [00:00<?, ?it/s]INFO:console:Starting to analyze C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Video metadata:
  Overall # of frames:    3597
  Duration of video [s]:  119.90
  fps:                    30.0
  resolution:             w=400, h=400

INFO:console:Running pose prediction with batch size 8
100%|██████████████████████████████████████████████████████████████████████████████| 3597/3597 [00:49<00:00, 72.91it/s]
INFO:console:Saving results in C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_cropDLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100.h5 and C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_cropDLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100_full.pickle
INFO:console:The videos are analyzed. Now your research can truly start!
You can create labeled videos with 'create_labeled_video'.
If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames.

INFO:console:Filtering with median model C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Saving filtered csv poses!
100%|█████████████████████████████████████████████████████████████████████████████| 3597/3597 [00:07<00:00, 512.71it/s]
INFO:console:Starting to process video: C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:Loading C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi and data.
INFO:console:Duration of video [s]: 119.9, recorded with 30.0 fps!
INFO:console:Overall # of frames: 3597 with cropped frame dimensions: 400 400
INFO:console:Generating frames and creating video.
INFO:console:Labeled videos created.
INFO:console:Loading
INFO:console:
INFO:console:C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\videos\DSC_3231_roi_#1_crop.avi
INFO:console:
INFO:console:and data.
INFO:console:Plots created! Please check the directory "plot-poses" within the video directory
```
</details>

[Evaluation-results.csvの解釈](https://deepwiki.com/DeepLabCut/DeepLabCut/2.5-model-evaluation)
について


bodypartsごとのrmseの評価は`comparisonbodyparts`を個別指定しないといけなさそう。

```py
deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\config.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['manubrium_base', 'mouth', 'prey', 'tentacle_base_1', 'tentacle_base_2', 'tentacle_base_3', 'tentacle_base_4', 'tentacle_base_5', 'tentacle_base_6', 'tentacle_base_7', 'tentacle_base_8', 'tentacle_base_9'])
```
<details><summary> bodypartsごとのevaluation結果
</summary>

```py
In [5]: deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\co
   ...: nfig.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['manubrium_base', 'mouth', 'prey', 'tentacle_bas
   ...: e_1', 'tentacle_base_2', 'tentacle_base_3', 'tentacle_base_4', 'tentacle_base_5', 'tentacle_base_6', 'tentacle_
   ...: base_7', 'tentacle_base_8', 'tentacle_base_9'])
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:12<00:00, 52.18it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 56.71it/s]
Evaluation results for DLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100-results.csv (pcutoff: 0.6):
train rmse             1.72
train rmse_pcutoff     1.40
train mAP             98.00
train mAR             98.67
test rmse              4.31
test rmse_pcutoff      3.24
test mAP              93.05
test mAR              93.86
Name: (0.9, 2, 100, -1, 0.6), dtype: float64

In [6]: deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\co
   ...: nfig.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['manubrium_base'])
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:11<00:00, 54.22it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 54.75it/s]
Evaluation results for DLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100-results.csv (pcutoff: 0.6):
train rmse            1.41
train rmse_pcutoff    1.37
train mAP             0.00
train mAR             0.00
test rmse             2.32
test rmse_pcutoff     1.88
test mAP              0.00
test mAR              0.00
Name: (0.9, 2, 100, -1, 0.6), dtype: float64

In [7]: deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\co
   ...: nfig.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['mouth'])
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:11<00:00, 54.28it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 55.76it/s]
Evaluation results for DLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100-results.csv (pcutoff: 0.6):
train rmse            1.96
train rmse_pcutoff    1.51
train mAP             0.00
train mAR             0.00
test rmse             2.88
test rmse_pcutoff     2.43
test mAP              0.00
test mAR              0.00
Name: (0.9, 2, 100, -1, 0.6), dtype: float64

In [8]: deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\co
   ...: nfig.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['prey'])
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:11<00:00, 53.77it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 56.71it/s]
Evaluation results for DLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100-results.csv (pcutoff: 0.6):
train rmse            2.33
train rmse_pcutoff    1.39
train mAP             0.00
train mAR             0.00
test rmse             2.46
test rmse_pcutoff     2.23
test mAP              0.00
test mAR              0.00
Name: (0.9, 2, 100, -1, 0.6), dtype: float64

In [9]: deeplabcut.evaluate_network(r'C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\co
   ...: nfig.yaml', Shuffles=[2], plotting=False, comparisonbodyparts=['tentacle_base_1', 'tentacle_base_2', 'tentacle_
   ...: base_3', 'tentacle_base_4', 'tentacle_base_5', 'tentacle_base_6', 'tentacle_base_7', 'tentacle_base_8', 'tentac
   ...: le_base_9'])
100%|████████████████████████████████████████████████████████████████████████████████| 630/630 [00:11<00:00, 54.74it/s]
100%|██████████████████████████████████████████████████████████████████████████████████| 70/70 [00:01<00:00, 55.32it/s]
Evaluation results for DLC_Resnet50_Cladonema_starved_cropJun20shuffle2_snapshot_100-results.csv (pcutoff: 0.6):
train rmse             1.70
train rmse_pcutoff     1.39
train mAP             97.88
train mAR             98.48
test rmse              4.68
test rmse_pcutoff      3.42
test mAP              92.39
test mAR              93.14
Name: (0.9, 2, 100, -1, 0.6), dtype: float64
```

</details>

- manubrium_base: train rmse            1.41
- mouth: train rmse            1.96
- prey: train rmse            2.33
- tentacle_base: train rmse             1.70


## 25/06/26

### Task after meeting
1. tentacle_baseを1~20に増やして再ラベル
2. tentacle_baseのlabelの仕方を変更。真下を1とか
3. training phaseでbodypartごとのerror histogramを出力
4. test phaseでbodypartごとのlikelihood histogramを出力
5. test phaseでの各maximum errorを出力
6. liklihood histogramをもとにthreshold(p-cutoff)を設定

**Done**
- anaconda上に[hdf5view](https://tgwoodcock.github.io/hdf5view/user/index.html)をinstall
  ->.h5ファイルを読めるように
- for task1&2: 新しいproject`Cladonema_starved_crop_tentacle20-Izuki-2025-06-26`を作成
  - `Cladonema_starved_crop-Vlad&Genta-2025-06-20`からextracted frameをコピー。使えるlabeled frameは残した
  - "tentacle_base1"は画像の一番下に位置するものと定める
  - config_path= `"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop_tentacle20-Izuki-2025-06-26\config.yaml"`
- for task4&5: .h5 fileの使い方を勉強し、計算用に`for_h5_trial.ipynb`を作成
  - pass `"C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_crop-Vlad&Genta-2025-06-20\evaluation-results-pytorch\iteration-0\Cladonema_starved_cropJun20-trainset95shuffle1\for_h5_trial.ipynb"`


## 25/06/27

**Done**
- LG gramにanacondaとpythonのpathを通して解析できるようにした


## 25/06/28

**Done**
- DSC_3184_#2までlabel済み
- 280 framesを90% train、10% testに回してtraining network
- .h5をdataframeにして計算し、excelにbodypartごとに出力するipynbができた


## 25/06/30

**Done**
- 280 framesをtraining&testした結果が出た。200epochs回すはずが151epochsで止まっていたけど
  <details><summary> evaluationの結果
  </summary>
  ```py
  100%|████████████████████████████████████████████████████████████████████████████████| 252/252 [00:06<00:00, 37.71it/s]
  00%|██████████████████████████████████████████████████████████████████████████████████| 28/28 [00:00<00:00, 46.67it/s]
  Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\gui\tabs\evaluate_network.py", line 207, in evaluate_network
    deeplabcut.evaluate_network(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\compat.py", line 558, in evaluate_network
    return evaluate_network(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 832, in evaluate_network
    evaluate_snapshot(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 629, in evaluate_snapshot
    save_evaluation_results(df_scores, scores_filepath, show_errors, pcutoff)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 881, in save_evaluation_results
    df_scores.to_csv(scores_path)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\util\_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\core\generic.py", line 3986, in to_csv
    return DataFrameRenderer(formatter).to_csv(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\formats\format.py", line 1014, in to_csv
    csv_formatter.save()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\formats\csvs.py", line 251, in save
    with get_handle(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
  FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\satie\\Desktop\\izuki_temp\\Cladonema_starved_crop_tentacle20-Izuki-2025-06-26\\evaluation-results-pytorch\\iteration-0\\Cladonema_starved_crop_tentacle20Jun26-trainset90shuffle1\\DLC_Resnet50_Cladonema_starved_crop_tentacle20Jun26shuffle1_snapshot_040-results.csv'
  INFO:console:Evaluation results for DLC_Resnet50_Cladonema_starved_crop_tentacle20Jun26shuffle1_snapshot_040-results.csv (pcutoff: 0.6):
  INFO:console:train rmse              2.00
  train rmse_pcutoff      1.95
  train mAP              99.89
  train mAR              99.96
  test rmse               2.41
  test rmse_pcutoff       2.26
  test mAP              100.00
  test mAR              100.00
  Name: (0.9, 1, 40, -1, 0.6), dtype: float64
  ```
  </details>

- evaluationでエラー吐く
  <details><summary> evaluationのエラー
  </summary>
  ```py
  100%|████████████████████████████████████████████████████████████████████████████████| 252/252 [00:05<00:00, 50.30it/s]
  100%|██████████████████████████████████████████████████████████████████████████████████| 28/28 [00:00<00:00, 52.39it/s]
  Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\gui\tabs\evaluate_network.py", line 207, in evaluate_network
    deeplabcut.evaluate_network(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\compat.py", line 558, in evaluate_network
    return evaluate_network(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 832, in evaluate_network
    evaluate_snapshot(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 629, in evaluate_snapshot
    save_evaluation_results(df_scores, scores_filepath, show_errors, pcutoff)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\pose_estimation_pytorch\apis\evaluation.py", line 881, in save_evaluation_results
    df_scores.to_csv(scores_path)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\util\_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\core\generic.py", line 3986, in to_csv
    return DataFrameRenderer(formatter).to_csv(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\formats\format.py", line 1014, in to_csv
    csv_formatter.save()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\formats\csvs.py", line 251, in save
    with get_handle(
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
  FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\satie\\Desktop\\izuki_temp\\Cladonema_starved_crop_tentacle20-Izuki-2025-06-26\\evaluation-results-pytorch\\iteration-0\\Cladonema_starved_crop_tentacle20Jun26-trainset90shuffle1\\DLC_Resnet50_Cladonema_starved_crop_tentacle20Jun26shuffle1_snapshot_040-results.csv'
  ```
  </details>
  ファイル名の長さが問題？


## 25/07/01

**Done**
- [このサイト](https://office54.net/iot/windows11/network-drive-map-release)に従い、ネットワークドライブの割り当てを行いファイルパスを短くした。その結果保存されなかったanalyze後のcsvが保存された
  <- ログアウト時に解除する習慣をつける
  network path:`\\DESKTOP-4DEN2DA\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26`
- 合計で540frame分labelし直し、trainingを行った(shuffle3)
- 時間で切り出したfed movie(~DSC_3201_#5)を400×400にcrop
  - 後でstarvedでtrainしたmodelに食わせる
  - 結果次第ではtraining dataにfedを入れる
- DSC_3204_#1に対してshuffle1~3を適用(数字が大きいほどlabeled dataが多い)
  - shuffle1: 280frames, 48m
  - shuffle2: 380frames, 50m
  - shuffle3: 540frames, 105m
- `Create videos`で`plot trajectory`にチェックを入れるとグラフを4つ出してくれる。
  colorを`hsv`にしているがこれだとmanubrium_baseとtentacle_base_#lastの色が被ってよろしくない。
- 明日はfedのanalyzeとanalyze結果vs私のlabelで精度計算


## 25/07/02

**Done**
- shuffle1~3で使用したframeを確認するために`Z:\evaluation-results-pytorch\iteration-0`内の.h5ファイルをcsv/excelに変換する.pyを作成
- `compare_label.xlsx`を作って、学習データに使わなかった動画に対しdlcが打ったラベルと私のラベルを比較できるようにした
- config.yamlを書き換えてたらよくわからんエラーが多発するようになった。泣きたい
  - napariでlabel済みファイルが開けない
  - 開けてもlabelが消えてる
  - あるはずのshuffleが見えない
  - アプリが勝手に落ちる etc...

<details><summary> 謎エラーーーー
</summary>

```py
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1202, in ViewerModel._open_or_raise_error(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], kwargs={}, layer_type=None, stack=False)
   1201 try:
-> 1202     added = self._add_layers_with_plugins(
        self = Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoom=1.0, angles=(0.0, 0.0, 90.0), perspective=0.0, mouse_pan=True, mouse_zoom=True), cursor=Cursor(position=(1.0, 1.0), scaled=True, size=1, style=<CursorStyle.STANDARD: 'standard'>), dims=Dims(ndim=2, ndisplay=2, last_used=0, range=((0, 2, 1), (0, 2, 1)), current_step=(0, 0), order=(0, 1), axis_labels=('0', '1')), grid=GridCanvas(stride=1, shape=(-1, -1), enabled=False), layers=[], help='', status='Ready', tooltip=Tooltip(visible=True, text=''), theme='dark', title='napari', mouse_over_canvas=True, mouse_move_callbacks=[], mouse_drag_callbacks=[], mouse_double_click_callbacks=[], mouse_wheel_callbacks=[<function dims_scroll at 0x00000207F531CDC0>], _persisted_mouse_event={}, _mouse_drag_gen={}, _mouse_wheel_gen={}, keymap={})
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        kwargs = {}
        stack = False
        plugin = 'napari-deeplabcut'
        layer_type = None
   1203         paths,
   1204         kwargs=kwargs,
   1205         stack=stack,
   1206         plugin=plugin,
   1207         layer_type=layer_type,
   1208     )
   1209 # plugin failed

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1292, in ViewerModel._add_layers_with_plugins(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, kwargs={}, plugin='napari-deeplabcut', layer_type=None)
   1291     assert len(paths) == 1
-> 1292     layer_data, hookimpl = read_data_with_plugins(
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        stack = False
        plugin = 'napari-deeplabcut'
   1293         paths, plugin=plugin, stack=stack
   1294     )
   1296 # glean layer names from filename. These will be used as *fallback*
   1297 # names, if the plugin does not return a name kwarg in their meta dict.

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\plugins\io.py:77, in read_data_with_plugins(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], plugin='napari-deeplabcut', stack=False)
     75 hookimpl: Optional[HookImplementation]
---> 77 res = _npe2.read(paths, plugin, stack=stack)
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        plugin = 'napari-deeplabcut'
        stack = False
        _npe2 = <module 'napari.plugins._npe2' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\napari\\plugins\\_npe2.py'>
     78 if res is not None:

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\plugins\_npe2.py:63, in read(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], plugin='napari-deeplabcut', stack=False)
     62 try:
---> 63     layer_data, reader = io_utils.read_get_reader(
        io_utils = <module 'npe2.io_utils' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\npe2\\io_utils.py'>
        plugin = 'napari-deeplabcut'
        npe1_path = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
     64         npe1_path, plugin_name=plugin
     65     )
     66 except ValueError as e:
     67     # plugin wasn't passed and no reader was found

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\io_utils.py:66, in read_get_reader(path=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5', plugin_name='napari-deeplabcut', stack=None)
     65     new_path, new_stack = v1_to_v2(path)
---> 66     return _read(
        new_path = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        new_stack = False
        plugin_name = 'napari-deeplabcut'
     67         new_path, plugin_name=plugin_name, return_reader=True, stack=new_stack
     68     )
     69 else:

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\io_utils.py:170, in _read(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, plugin_name='napari-deeplabcut', return_reader=True, _pm=<npe2._plugin_manager.PluginManager object>)
    168 if read_func is not None:
    169     # if the reader function raises an exception here, we don't try to catch it
--> 170     if layer_data := read_func(paths, stack=stack):
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        read_func = <function read_hdf at 0x0000020822F66830>
        stack = False
    171         return (layer_data, rdr) if return_reader else layer_data

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\manifest\contributions\_readers.py:69, in ReaderContribution.exec.<locals>.npe1_compat(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False)
     68 path = v2_to_v1(paths, stack)
---> 69 return callable_(path)
        path = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
        callable_ = <function read_hdf at 0x000002081C504940>

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari_deeplabcut\_reader.py:201, in read_hdf(filename=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5')
    200 for filename in glob.iglob(filename):
--> 201     temp = pd.read_hdf(filename)
        filename = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
        pd = <module 'pandas' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\pandas\\__init__.py'>
    202     temp = misc.merge_multiple_scorers(temp)

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:457, in read_hdf(path_or_buf=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5', key='/keypoints', mode='r', errors='strict', where=None, start=None, stop=None, columns=None, iterator=False, chunksize=None, **kwargs={})
    456         key = candidate_only_group._v_pathname
--> 457     return store.select(
        store = <class 'pandas.io.pytables.HDFStore'>
File path: D:\Satiety_differentially_modulates_feeding_\dlc\Cladonema_starved_tentacle20-Izuki-2025-06-26\original_labeled-data\DSC_3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5

        key = '/keypoints'
        where = None
        start = None
        stop = None
        columns = None
        iterator = False
        chunksize = None
        auto_close = True
    458         key,
    459         where=where,
    460         start=start,
    461         stop=stop,
    462         columns=columns,
    463         iterator=iterator,
    464         chunksize=chunksize,
    465         auto_close=auto_close,
    466     )
    467 except (ValueError, TypeError, LookupError):

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:911, in HDFStore.select(self=<class 'pandas.io.pytables.HDFStore'>
File path:...3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5
, key='/keypoints', where=None, start=None, stop=None, columns=None, iterator=False, chunksize=None, auto_close=True)
    898 it = TableIterator(
    899     self,
    900     s,
   (...)
    908     auto_close=auto_close,
    909 )
--> 911 return it.get_result()
        it = <pandas.io.pytables.TableIterator object at 0x0000020822B7C6D0>

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:2034, in TableIterator.get_result(self=<pandas.io.pytables.TableIterator object>, coordinates=False)
   2033 # directly return the result
-> 2034 results = self.func(self.start, self.stop, where)
        self = <pandas.io.pytables.TableIterator object at 0x0000020822B7C6D0>
        where = None
        self.start = None
        self.stop = None
   2035 self.close()

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:895, in HDFStore.select.<locals>.func(_start=None, _stop=None, _where=None)
    894 def func(_start, _stop, _where):
--> 895     return s.read(start=_start, stop=_stop, where=_where, columns=columns)
        Exception trying to inspect frame. No more locals available.

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:3301, in BlockManagerFixed.read(self=<class 'pandas.io.pytables.FrameFixed'> instance, where=None, columns=None, start=None, stop=None)
   3300 _start, _stop = (start, stop) if i == select_axis else (None, None)
-> 3301 ax = self.read_index(f"axis{i}", start=_start, stop=_stop)
        i = 0
        Exception trying to inspect frame. No more locals available.
   3302 axes.append(ax)

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\pandas\io\pytables.py:2991, in GenericFixed.read_index(self=<class 'pandas.io.pytables.FrameFixed'> instance, key='axis0', start=None, stop=None)
   2988 def read_index(
   2989     self, key: str, start: int | None = None, stop: int | None = None
   2990 ) -> Index:
-> 2991     variety = _ensure_decoded(getattr(self.attrs, f"{key}_variety"))
        key = 'axis0'
        Exception trying to inspect frame. No more locals available.
   2993     if variety == "multi":

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\tables\attributeset.py:283, in AttributeSet.__getattr__(self=/keypoints._v_attrs (AttributeSet), 8 attributes..._type := 'frame',
    pandas_version := '0.15.2'], name='axis0_variety')
    282 if name not in self._v_attrnames:
--> 283     raise AttributeError(f"Attribute {name!r} does not exist "
        name = 'axis0_variety'
        self = /keypoints._v_attrs (AttributeSet), 8 attributes:
   [CLASS := 'GROUP',
    TITLE := '',
    VERSION := '1.0',
    encoding := 'UTF-8',
    errors := 'strict',
    ndim := 2,
    pandas_type := 'frame',
    pandas_version := '0.15.2']
        self._v__nodepath = '/keypoints'
    284                          f"in node: {self._v__nodepath!r}")
    286 # Read the attribute from disk. This is an optimization to read
    287 # quickly system attributes that are _string_ values, but it
    288 # takes care of other types as well as for example NROWS for
    289 # Tables and EXTDIM for EArrays

AttributeError: Attribute 'axis0_variety' does not exist in node: '/keypoints'

The above exception was the direct cause of the following exception:

ReaderPluginError                         Traceback (most recent call last)
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\_qt\qt_viewer.py:953, in QtViewer._qt_open(self=<napari._qt.qt_viewer.QtViewer object>, filenames=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, choose_plugin=False, plugin=None, layer_type=None, **kwargs={})
    952 try:
--> 953     self.viewer.open(
        self = <napari._qt.qt_viewer.QtViewer object at 0x00000207F67395A0>
        self.viewer = Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoom=1.0, angles=(0.0, 0.0, 90.0), perspective=0.0, mouse_pan=True, mouse_zoom=True), cursor=Cursor(position=(1.0, 1.0), scaled=True, size=1, style=<CursorStyle.STANDARD: 'standard'>), dims=Dims(ndim=2, ndisplay=2, last_used=0, range=((0, 2, 1), (0, 2, 1)), current_step=(0, 0), order=(0, 1), axis_labels=('0', '1')), grid=GridCanvas(stride=1, shape=(-1, -1), enabled=False), layers=[], help='', status='Ready', tooltip=Tooltip(visible=True, text=''), theme='dark', title='napari', mouse_over_canvas=True, mouse_move_callbacks=[], mouse_drag_callbacks=[], mouse_double_click_callbacks=[], mouse_wheel_callbacks=[<function dims_scroll at 0x00000207F531CDC0>], _persisted_mouse_event={}, _mouse_drag_gen={}, _mouse_wheel_gen={}, keymap={})
        filenames = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        stack = False
        plugin = None
        layer_type = None
        kwargs = {}
    954         filenames,
    955         stack=stack,
    956         plugin=plugin,
    957         layer_type=layer_type,
    958         **kwargs,
    959     )
    960 except ReaderPluginError as e:

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1102, in ViewerModel.open(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), path=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, plugin=None, layer_type=None, **kwargs={})
   1100 # no plugin choice was made
   1101 else:
-> 1102     layers = self._open_or_raise_error(
        layers = <module 'napari.layers' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\napari\\layers\\__init__.py'>
        self = Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoom=1.0, angles=(0.0, 0.0, 90.0), perspective=0.0, mouse_pan=True, mouse_zoom=True), cursor=Cursor(position=(1.0, 1.0), scaled=True, size=1, style=<CursorStyle.STANDARD: 'standard'>), dims=Dims(ndim=2, ndisplay=2, last_used=0, range=((0, 2, 1), (0, 2, 1)), current_step=(0, 0), order=(0, 1), axis_labels=('0', '1')), grid=GridCanvas(stride=1, shape=(-1, -1), enabled=False), layers=[], help='', status='Ready', tooltip=Tooltip(visible=True, text=''), theme='dark', title='napari', mouse_over_canvas=True, mouse_move_callbacks=[], mouse_drag_callbacks=[], mouse_double_click_callbacks=[], mouse_wheel_callbacks=[<function dims_scroll at 0x00000207F531CDC0>], _persisted_mouse_event={}, _mouse_drag_gen={}, _mouse_wheel_gen={}, keymap={})
        _path = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        _stack = False
        kwargs = {}
        layer_type = None
   1103         _path, kwargs, layer_type, _stack
   1104     )
   1105     added.extend(layers)

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1211, in ViewerModel._open_or_raise_error(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], kwargs={}, layer_type=None, stack=False)
   1210     except Exception as e:  # noqa: BLE001
-> 1211         raise ReaderPluginError(
        trans = <napari.utils.translations.TranslationBundle object at 0x00000207EFFDC910>
        plugin = 'napari-deeplabcut'
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
   1212             trans._(
   1213                 'Tried opening with {plugin}, but failed.',
   1214                 deferred=True,
   1215                 plugin=plugin,
   1216             ),
   1217             plugin,
   1218             paths,
   1219         ) from e
   1220 # multiple plugins
   1221 else:

ReaderPluginError: Tried opening with napari-deeplabcut, but failed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\_qt\qt_viewer.py:1318, in QtViewer.dropEvent(self=<napari._qt.qt_viewer.QtViewer object>, event=<PyQt5.QtGui.QDropEvent object>)
   1315     else:
   1316         filenames.append(url.toString())
-> 1318 self._qt_open(
        self = <napari._qt.qt_viewer.QtViewer object at 0x00000207F67395A0>
        filenames = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        shift_down = <PyQt5.QtCore.Qt.KeyboardModifiers object at 0x000002081EB57990>
        alt_down = <PyQt5.QtCore.Qt.KeyboardModifiers object at 0x0000020822587300>
   1319     filenames,
   1320     stack=bool(shift_down),
   1321     choose_plugin=bool(alt_down),
   1322 )

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\_qt\qt_viewer.py:961, in QtViewer._qt_open(self=<napari._qt.qt_viewer.QtViewer object>, filenames=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, choose_plugin=False, plugin=None, layer_type=None, **kwargs={})
    953     self.viewer.open(
    954         filenames,
    955         stack=stack,
   (...)
    958         **kwargs,
    959     )
    960 except ReaderPluginError as e:
--> 961     handle_gui_reading(
        filenames = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        self = <napari._qt.qt_viewer.QtViewer object at 0x00000207F67395A0>
        stack = False
        layer_type = None
        kwargs = {}
    962         filenames,
    963         self,
    964         stack,
    965         e.reader_plugin,
    966         e,
    967         layer_type=layer_type,
    968         **kwargs,
    969     )
    970 except MultipleReaderError:
    971     handle_gui_reading(filenames, self, stack, **kwargs)

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\_qt\dialogs\qt_reader_dialog.py:201, in handle_gui_reading(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], qt_viewer=<napari._qt.qt_viewer.QtViewer object>, stack=False, plugin_name='napari-deeplabcut', error=ReaderPluginError('Tried opening with napari-deeplabcut, but failed.'), plugin_override=False, **kwargs={'layer_type': None})
    199 display_name, persist = readerDialog.get_user_choices()
    200 if display_name:
--> 201     open_with_dialog_choices(
        display_name = 'napari builtins'
        persist = False
        readerDialog = <napari._qt.dialogs.qt_reader_dialog.QtReaderDialog object at 0x0000020822F67E20>
        readerDialog._extension = '.h5'
        readers = {'napari': 'napari builtins'}
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        stack = False
        qt_viewer = <napari._qt.qt_viewer.QtViewer object at 0x00000207F67395A0>
        kwargs = {'layer_type': None}
    202         display_name,
    203         persist,
    204         readerDialog._extension,
    205         readers,
    206         paths,
    207         stack,
    208         qt_viewer,
    209         **kwargs,
    210     )

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\_qt\dialogs\qt_reader_dialog.py:294, in open_with_dialog_choices(display_name='napari builtins', persist=False, extension='.h5', readers={'napari': 'napari builtins'}, paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, qt_viewer=<napari._qt.qt_viewer.QtViewer object>, **kwargs={'layer_type': None})
    290 plugin_name = [
    291     p_name for p_name, d_name in readers.items() if d_name == display_name
    292 ][0]
    293 # may throw error, but we let it this time
--> 294 qt_viewer.viewer.open(paths, stack=stack, plugin=plugin_name, **kwargs)
        plugin_name = 'napari'
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        qt_viewer.viewer = Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoom=1.0, angles=(0.0, 0.0, 90.0), perspective=0.0, mouse_pan=True, mouse_zoom=True), cursor=Cursor(position=(1.0, 1.0), scaled=True, size=1, style=<CursorStyle.STANDARD: 'standard'>), dims=Dims(ndim=2, ndisplay=2, last_used=0, range=((0, 2, 1), (0, 2, 1)), current_step=(0, 0), order=(0, 1), axis_labels=('0', '1')), grid=GridCanvas(stride=1, shape=(-1, -1), enabled=False), layers=[], help='', status='Ready', tooltip=Tooltip(visible=True, text=''), theme='dark', title='napari', mouse_over_canvas=True, mouse_move_callbacks=[], mouse_drag_callbacks=[], mouse_double_click_callbacks=[], mouse_wheel_callbacks=[<function dims_scroll at 0x00000207F531CDC0>], _persisted_mouse_event={}, _mouse_drag_gen={}, _mouse_wheel_gen={}, keymap={})
        stack = False
        kwargs = {'layer_type': None}
        qt_viewer = <napari._qt.qt_viewer.QtViewer object at 0x00000207F67395A0>
    296 if persist:
    297     if not extension.endswith(os.sep):

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1092, in ViewerModel.open(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), path=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, plugin='napari', layer_type=None, **kwargs={})
   1089 _path = [_path] if not isinstance(_path, list) else _path
   1090 if plugin:
   1091     added.extend(
-> 1092         self._add_layers_with_plugins(
        added = []
        self = Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoom=1.0, angles=(0.0, 0.0, 90.0), perspective=0.0, mouse_pan=True, mouse_zoom=True), cursor=Cursor(position=(1.0, 1.0), scaled=True, size=1, style=<CursorStyle.STANDARD: 'standard'>), dims=Dims(ndim=2, ndisplay=2, last_used=0, range=((0, 2, 1), (0, 2, 1)), current_step=(0, 0), order=(0, 1), axis_labels=('0', '1')), grid=GridCanvas(stride=1, shape=(-1, -1), enabled=False), layers=[], help='', status='Ready', tooltip=Tooltip(visible=True, text=''), theme='dark', title='napari', mouse_over_canvas=True, mouse_move_callbacks=[], mouse_drag_callbacks=[], mouse_double_click_callbacks=[], mouse_wheel_callbacks=[<function dims_scroll at 0x00000207F531CDC0>], _persisted_mouse_event={}, _mouse_drag_gen={}, _mouse_wheel_gen={}, keymap={})
        _path = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        kwargs = {}
        plugin = 'napari'
        layer_type = None
        _stack = False
   1093             _path,
   1094             kwargs=kwargs,
   1095             plugin=plugin,
   1096             layer_type=layer_type,
   1097             stack=_stack,
   1098         )
   1099     )
   1100 # no plugin choice was made
   1101 else:
   1102     layers = self._open_or_raise_error(
   1103         _path, kwargs, layer_type, _stack
   1104     )

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\components\viewer_model.py:1292, in ViewerModel._add_layers_with_plugins(self=Viewer(camera=Camera(center=(0.0, 0.0, 0.0), zoo...ouse_drag_gen={}, _mouse_wheel_gen={}, keymap={}), paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, kwargs={}, plugin='napari', layer_type=None)
   1290 else:
   1291     assert len(paths) == 1
-> 1292     layer_data, hookimpl = read_data_with_plugins(
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        stack = False
        plugin = 'napari'
   1293         paths, plugin=plugin, stack=stack
   1294     )
   1296 # glean layer names from filename. These will be used as *fallback*
   1297 # names, if the plugin does not return a name kwarg in their meta dict.
   1298 filenames = []

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\plugins\io.py:77, in read_data_with_plugins(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], plugin='napari', stack=False)
     74     assert len(paths) == 1
     75 hookimpl: Optional[HookImplementation]
---> 77 res = _npe2.read(paths, plugin, stack=stack)
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        plugin = 'napari'
        stack = False
        _npe2 = <module 'napari.plugins._npe2' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\napari\\plugins\\_npe2.py'>
     78 if res is not None:
     79     _ld, hookimpl = res

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari\plugins\_npe2.py:63, in read(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], plugin='napari', stack=False)
     61     npe1_path = paths[0]
     62 try:
---> 63     layer_data, reader = io_utils.read_get_reader(
        io_utils = <module 'npe2.io_utils' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\npe2\\io_utils.py'>
        plugin = 'napari'
        npe1_path = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
     64         npe1_path, plugin_name=plugin
     65     )
     66 except ValueError as e:
     67     # plugin wasn't passed and no reader was found
     68     if 'No readers returned data' not in str(e):

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\io_utils.py:66, in read_get_reader(path=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5', plugin_name='napari', stack=None)
     62 if stack is None:
     63     # "npe1" old path
     64     # Napari 0.4.15 and older, hopefully we can drop this and make stack mandatory
     65     new_path, new_stack = v1_to_v2(path)
---> 66     return _read(
        new_path = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        new_stack = False
        plugin_name = 'napari'
     67         new_path, plugin_name=plugin_name, return_reader=True, stack=new_stack
     68     )
     69 else:
     70     assert isinstance(path, list)

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\io_utils.py:170, in _read(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False, plugin_name='napari', return_reader=True, _pm=<npe2._plugin_manager.PluginManager object>)
    165     read_func = rdr.exec(
    166         kwargs={"path": paths, "stack": stack, "_registry": _pm.commands}
    167     )
    168     if read_func is not None:
    169         # if the reader function raises an exception here, we don't try to catch it
--> 170         if layer_data := read_func(paths, stack=stack):
        paths = ['D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5']
        read_func = <function _magic_imreader at 0x0000020822F669E0>
        stack = False
    171             return (layer_data, rdr) if return_reader else layer_data
    173 if plugin_name:

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\npe2\manifest\contributions\_readers.py:69, in ReaderContribution.exec.<locals>.npe1_compat(paths=[r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5'], stack=False)
     66 @wraps(callable_)
     67 def npe1_compat(paths, *, stack):
     68     path = v2_to_v1(paths, stack)
---> 69     return callable_(path)
        path = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
        callable_ = <function _magic_imreader at 0x00000207F0925240>

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari_builtins\io\_read.py:491, in _magic_imreader(path=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5')
    490 def _magic_imreader(path: str) -> List["LayerData"]:
--> 491     return [(magic_imread(path),)]
        path = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari_builtins\io\_read.py:227, in magic_imread(filenames=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5', use_dask=False, stack=True)
    225 else:
    226     if shape is None:
--> 227         image = imread(filename)
        filename = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
    228         shape = image.shape
    229         dtype = image.dtype

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari_builtins\io\_read.py:94, in imread(filename=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5')
     92     return np.load(filename)
     93 if ext.lower() not in [".tif", ".tiff", ".lsm"]:
---> 94     return imageio.imread(filename)
        filename = 'D:\\Satiety_differentially_modulates_feeding_\\dlc\\Cladonema_starved_tentacle20-Izuki-2025-06-26\\original_labeled-data\\DSC_3174_roi_trial1_crop\\CollectedData_Vlad&Genta.h5'
        imageio = <module 'imageio.v2' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\imageio\\v2.py'>
     95 import tifffile
     97 # Pre-download urls before loading them with tifffile

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\imageio\v2.py:360, in imread(uri=r'D:\Satiety_differentially_modulates_feeding_\dlc..._3174_roi_trial1_crop\CollectedData_Vlad&Genta.h5', format=None, **kwargs={})
    357 imopen_args["legacy_mode"] = True
    359 with imopen(uri, "ri", **imopen_args) as file:
--> 360     result = file.read(index=0, **kwargs)
        file = <imageio.plugins.pillow.PillowPlugin object at 0x0000020822B7D030>
        kwargs = {}
    362 return result

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\imageio\plugins\pillow.py:252, in PillowPlugin.read(self=<imageio.plugins.pillow.PillowPlugin object>, index=0, mode=None, rotate=False, apply_gamma=False, writeable_output=True, pilmode=None, exifrotate=None, as_gray=None)
    249 if isinstance(index, int):
    250     # will raise IO error if index >= number of frames in image
    251     self._image.seek(index)
--> 252     image = self._apply_transforms(
        self = <imageio.plugins.pillow.PillowPlugin object at 0x0000020822B7D030>
        mode = None
        rotate = False
        self._image = <PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1 at 0x20822B7D180>
        apply_gamma = False
        writeable_output = True
    253         self._image, mode, rotate, apply_gamma, writeable_output
    254     )
    255 else:
    256     iterator = self.iter(
    257         mode=mode,
    258         rotate=rotate,
    259         apply_gamma=apply_gamma,
    260         writeable_output=writeable_output,
    261     )

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\imageio\plugins\pillow.py:333, in PillowPlugin._apply_transforms(self=<imageio.plugins.pillow.PillowPlugin object>, image=<PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1>, mode=None, rotate=False, apply_gamma=False, writeable_output=True)
    329     else:
    330         # pillow >= 10.1.0
    331         image = image.convert(desired_mode)
--> 333 image = np.asarray(image)
        image = <PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1 at 0x20822B7D180>
        np = <module 'numpy' from 'C:\\Users\\satie\\anaconda3\\envs\\DEEPLABCUT\\lib\\site-packages\\numpy\\__init__.py'>
    335 meta = self.metadata(index=self._image.tell(), exclude_applied=False)
    336 if rotate and "Orientation" in meta:

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\PIL\Image.py:735, in Image.__array_interface__(self=<PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1>)
    733     new["data"] = self.tobytes("raw", "L")
    734 else:
--> 735     new["data"] = self.tobytes()
        new = {'version': 3}
        self = <PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1 at 0x20822B7D180>
    736 new["shape"], new["typestr"] = _conv_type_shape(self)
    737 return new

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\PIL\Image.py:794, in Image.tobytes(self=<PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1>, encoder_name='raw', *args=())
    791 if encoder_name == "raw" and encoder_args == ():
    792     encoder_args = self.mode
--> 794 self.load()
        self = <PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1 at 0x20822B7D180>
    796 if self.width == 0 or self.height == 0:
    797     return b""

File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\PIL\ImageFile.py:473, in StubImageFile.load(self=<PIL.Hdf5StubImagePlugin.HDF5StubImageFile image mode=F size=1x1>)
    471 if loader is None:
    472     msg = f"cannot find loader for this {self.format} file"
--> 473     raise OSError(msg)
        msg = 'cannot find loader for this HDF5 file'
    474 image = loader.load(self)
    475 assert image is not None

OSError: cannot find loader for this HDF5 file
```
</details>


## 25/07/03

**Done**
- napariでファイルが開けないエラーを解消すべく.yamlファイルを触る
  ->ファイル名・フォルダ名・bodypartsの数変更あたりが悪さしている感じ
- labeled-dataを1個ずつ開いてみて開けるデータと開けないデータを把握
  - DSC_3147_#1~2, 3177_#2
  - ↑おそらく.h5 fileが破損していて、deleteしたらnapariで開けるようになった
- fed movie(DSC_3177_#2~3, DSC_3180_#3)を比較用にlabeling
  - 比較用のlabel(≠学習用のlabel)は正確性は多少犠牲にしても見えるところは全部点を打つようにする
  - fedだとpreyが底に落ちていたり、manubriumが大きくてmouthやbaseが追いにくいから学習データに入れたくないんだよな
- starvedのモデル(shuffle1,3)をfed movieに適用してみた
  - starved同様、隣のtentacleと混ざっちゃう。底に落ちてるpreyにも反応してそう
  - manubrium_baseがdetectできてない
- 本当はpredictionにおけるlikelihoodとlabelの対応関係/妥当性を見たかったけど、判断基準が見つからない。
  - `hist_filtered`を見る限り、50px以内に前後フレームの誤差は収まっていそう
- [NIBBのAI解析室が出している解説](https://aifacility.nibb.ac.jp/deeplabcut)を発見


## 25/07/04

**Done**
- likelihoodグラフからp-cutoffを決めるための参考資料を発見>[How do you decide what p-cutoff value is optimal?](https://forum.image.sc/t/basic-questions-on-the-user-guide/38975)
- [Deeplabcut-Wiki](https://deepwiki.com/DeepLabCut/DeepLabCut/2-project-lifecycle)発見


## 25/07/08

**Done**
- 比較用に`2025-06-26`model-shuffle3でDSC_3219_#5とDSC_3231_#1をanalyze
- ground truthを作るためDSC_3219_#5とDSC_3231_#1のlabeling
- `Extract outlier frames`の使い方を学んだ
- `2025-06-26`model-shuffle3でanalyzeしたfed movie(DSC_3177_#2~3, 3180_#3)で`Extract outlier frames`
  - outlieralgorithm = 'jump'
  - DSC_3177_#2~3からは20 framesずつ
  - DSC_3180_#3からはなぜか40 framesがextractされてた
- labelを修正したfed movieを`Merge dataset`し、shuffle4を作成
- Train network-shuffle4 ->Evaluate network
- fed movie(DSC_3203_#1~DSC_3213_#5)を400×400にcrop、enhance contrast(0.20)
- model-shuffle4で先ほどmergeしたfed movie(DSC_3177_#3)をreanalyze
- model-shuffle4でDSC_3191_#2, 4($\in$ training dataset, 10 tentacles)をreanalyze -> extract outlier frames -> refine label -> merge dataset
  - `Merge dataset`をすると元の20frame分が`Extract outlier frames`の20frameに置き換わった



<details><summary> extract outlier frames
</summary>

```py
  2272it [00:14, 155.98it/s]
  2241it [00:15, 148.42it/s]
  2245it [00:14, 156.99it/s]
  # ---中略---
  INFO:console:Method
  INFO:console:
  INFO:console:jump
  INFO:console:
  INFO:console: found
  INFO:console:
  INFO:console:2245
  INFO:console:
  INFO:console: putative outlier frames.
  INFO:console:Do you want to proceed with extracting
  INFO:console:
  INFO:console:20
  INFO:console:
  INFO:console: of those?
  INFO:console:If this list is very large, perhaps consider changing the parameters (start, stop, p_bound, comparisonbodyparts) or use a different method.
  INFO:console:Loading video...
  INFO:console:Cropping coords:
  INFO:console:
  INFO:console:None
  INFO:console:Duration of video [s]:
  INFO:console:
  INFO:console:89.93333333333334
  INFO:console:
  INFO:console:, recorded @
  INFO:console:
  INFO:console:30.0
  INFO:console:
  INFO:console:fps!
  INFO:console:Overall # of frames:
  INFO:console:
  INFO:console:2698
  INFO:console:
  INFO:console:with (cropped) frame dimensions:
  INFO:console:Kmeans-quantization based extracting of frames from
  INFO:console:
  INFO:console:0.0
  INFO:console:
  INFO:console: seconds to
  INFO:console:
  INFO:console:89.93
  INFO:console:
  INFO:console: seconds.
  INFO:console:Extracting and downsampling...
  INFO:console:
  INFO:console:2245
  INFO:console:
  INFO:console: frames from the video.
  INFO:console:Kmeans clustering ... (this might take a while)
  INFO:console:Let's select frames indices:
  INFO:console:
  INFO:console:[976, 2489, 1004, 1793, 1099, 1271, 1449, 1082, 2246, 568, 1249, 691, 1430, 340, 1602, 1035, 636, 823, 271, 540]
  INFO:console:Attempting to create a symbolic link of the video ...
  INFO:console:Created the symlink of C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\labeled-data_forAnalyze\DSC_3177_roi_trial2_crop.avi to C:\Users\satie\Desktop\izuki_temp\Cladonema_starved_tentacle20-Izuki-2025-06-26\videos\DSC_3177_roi_trial2_crop.avi
  INFO:console:New videos were added to the project! Use the function 'extract_frames' to select frames for labeling.
  INFO:console:The outlier frames are extracted. They are stored in the subdirectory labeled-data\DSC_3177_roi_trial2_crop.
  INFO:console:Once you extracted frames for all videos, use 'refine_labels' to manually correct the labels.
```
</details>

a